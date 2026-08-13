"""
BlindAssist — Real-Time Object Detection Voice Assistant (Local / VS Code)
============================================================================
Local counterpart of the Colab notebook version. Reads directly from your
webcam via OpenCV and speaks using offline TTS (pyttsx3) instead of the
browser's Web Speech API — no browser/Colab bridge involved.

SETUP (venv already created):
    pip install ultralytics opencv-python pyttsx3

    # Windows note: pyttsx3 uses SAPI5, works out of the box.
    # Linux note: pyttsx3 needs espeak installed: sudo apt install espeak

RUN:
    python main.py

Press 'q' in the video window (or Ctrl+C in the terminal) to stop.
"""

import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
import pyttsx3
import torch
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Config:
    model_path: str = "yolov8m.pt"
    confidence: float = 0.5
    inference_size: int = 640
    detection_interval: float = 0.8   # seconds between capture+detect cycles
    vote_window: int = 3              # detection passes considered per vote
    vote_threshold: int = 2           # label must appear in >= this many of the last `vote_window` passes
    camera_index: int = 0
    frame_width: int = 680
    frame_height: int = 480
    warmup_frames: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


class SpeechEngine:
    """Non-blocking wrapper around pyttsx3 so speaking never stalls the detection loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

    def _speak_sync(self, text: str) -> None:
        # pyttsx3 engines aren't thread-safe to share, so create one per call
        # and serialize with a lock to avoid overlapping/garbled speech.
        with self._lock:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()


class BlindAssistCamera:
    """Wraps a local webcam via OpenCV."""

    def __init__(self, config: Config):
        self.cfg = config
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        # CAP_DSHOW avoids OpenCV silently ignoring resolution requests on Windows;
        # harmless no-op on other platforms since we fall back if it fails to open.
        self.cap = cv2.VideoCapture(self.cfg.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.cfg.camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        # Keep only the newest frame in OpenCV's internal buffer. Without this,
        # any time our loop falls behind the camera's native frame rate (which
        # it will, since YOLO inference takes longer than one frame interval),
        # the buffer fills with stale frames and the feed looks laggy/delayed.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self.cfg.camera_index}. "
                "Check that it's connected and not in use by another app."
            )

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: requested {self.cfg.frame_width}x{self.cfg.frame_height}, "
              f"actual {actual_w}x{actual_h}")

        # Let auto-exposure/auto-focus settle before detection starts on real frames.
        for _ in range(self.cfg.warmup_frames):
            self.cap.read()

    def is_running(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.is_running():
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


class BlindAssistApp:
    """Detect -> vote for stability -> draw boxes -> announce arrivals/returns."""

    def __init__(self, config: Config):
        self.cfg = config
        self.camera = BlindAssistCamera(config)
        self.speech = SpeechEngine()
        print(f"Loading {config.model_path} on {config.device}...")
        self.model = YOLO(config.model_path)
        self.recent_passes: deque = deque(maxlen=config.vote_window)
        # Tracks what was present as of the *last* detection pass, so we can
        # tell arrivals/returns and departures apart from continuous presence.
        self.currently_present: Set[str] = set()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_boxes: List[Tuple[int, int, int, int, str, float]] = []
        self._frame_lock = threading.Lock()
        self._box_lock = threading.Lock()
        self._detector_stop = threading.Event()

    def _detect_objects(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        results = self.model(
            frame,
            conf=self.cfg.confidence,
            imgsz=self.cfg.inference_size,
            device=self.cfg.device,
            verbose=False,
        )
        names = self.model.names
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = names[int(box.cls[0])]
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, label, conf))
        return boxes

    def _stable_objects(self, current: Set[str]) -> Set[str]:
        """Only trust a label if it showed up in >= vote_threshold of the last
        vote_window detection passes. This absorbs single-frame flicker between
        similar-looking classes (e.g. bottle/banana/cup on an unfamiliar object)."""
        self.recent_passes.append(current)
        votes = Counter(label for pass_labels in self.recent_passes for label in pass_labels)
        return {label for label, count in votes.items() if count >= self.cfg.vote_threshold}

    def _announce_new_objects(self, stable_now: Set[str]) -> None:
        # Anything that wasn't present last pass but is now is either brand
        # new or has just walked back into frame.
        newly_arrived = stable_now - self.currently_present
        # Anything that was present last pass but isn't now has left frame.
        departed = self.currently_present - stable_now

        if newly_arrived:
            labels = sorted(newly_arrived)
            suffix = " detected" if len(labels) == 1 else "s detected"
            message = ", ".join(labels) + suffix
            print(f"[SPEAK] {message}")
            self.speech.speak(message)

        if departed:
            labels = sorted(departed)
            suffix = " gone" if len(labels) == 1 else "s gone"
            message = ", ".join(labels) + suffix
            print(f"[SPEAK] {message}")
            self.speech.speak(message)

        self.currently_present = stable_now

    @staticmethod
    def _draw_boxes(frame: np.ndarray, boxes: List[Tuple[int, int, int, int, str, float]]) -> np.ndarray:
        for x1, y1, x2, y2, label, conf in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return frame

    def _detection_worker(self) -> None:
        """Runs on its own thread so slow YOLO inference never blocks frame
        capture/display. Always grabs whatever the latest frame is, runs
        detection on it, then immediately looks for the newest frame again —
        effectively skipping any frames that arrived while it was busy."""
        last_detection_time = 0.0
        while not self._detector_stop.is_set():
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            if now - last_detection_time < self.cfg.detection_interval:
                time.sleep(0.01)
                continue

            boxes = self._detect_objects(frame)
            current_objects = {label for *_, label, _ in boxes}
            stable_now = self._stable_objects(current_objects)
            self._announce_new_objects(stable_now)

            with self._box_lock:
                self._latest_boxes = boxes

            last_detection_time = time.time()

    def run(self) -> None:
        self.camera.start()
        print("BlindAssist running. Press 'q' in the video window to stop.")

        detector_thread = threading.Thread(target=self._detection_worker, daemon=True)
        detector_thread.start()

        try:
            while self.camera.is_running():
                frame = self.camera.get_frame()
                if frame is None:
                    continue

                with self._frame_lock:
                    self._latest_frame = frame

                with self._box_lock:
                    boxes = self._latest_boxes

                display_frame = self._draw_boxes(frame.copy(), boxes)
                cv2.imshow("BlindAssist", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            self._detector_stop.set()
            detector_thread.join(timeout=2.0)
            self.camera.stop()
            print("BlindAssist stopped.")


if __name__ == "__main__":
    app = BlindAssistApp(CFG)
    app.run()
11