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
import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional, Set

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
    announce_cooldown: float = 6.0    # seconds before the same label can be announced again
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    warmup_frames: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    enable_distance_filter: bool = True
    max_distance_meters: float = 3.0   # ignore anything estimated farther than this
    camera_fov_degrees: float = 60.0   # typical laptop/webcam horizontal FOV — adjust if distances look off


# Approximate real-world height (meters) per COCO class, used for distance estimation.
# Unlisted classes fall back to DEFAULT_HEIGHT_M — a rough guess, so distance for
# those will be less accurate.
KNOWN_HEIGHTS_M = {
    "person": 1.7, "bicycle": 1.0, "car": 1.5, "motorcycle": 1.1, "bus": 3.2,
    "truck": 3.0, "bench": 0.9, "chair": 0.9, "couch": 0.8, "bed": 0.6,
    "dining table": 0.75, "tv": 0.5, "laptop": 0.25, "bottle": 0.25,
    "wine glass": 0.2, "cup": 0.1, "fork": 0.18, "knife": 0.2, "spoon": 0.16,
    "bowl": 0.1, "banana": 0.18, "apple": 0.08, "backpack": 0.45,
    "umbrella": 0.9, "handbag": 0.3, "suitcase": 0.6, "refrigerator": 1.7,
    "microwave": 0.3, "oven": 0.6, "sink": 0.2, "toilet": 0.4,
    "book": 0.25, "clock": 0.3, "vase": 0.3, "cell phone": 0.15,
    "keyboard": 0.03, "mouse": 0.04, "remote": 0.15, "scissors": 0.2,
    "toothbrush": 0.18, "hair drier": 0.25, "dog": 0.5, "cat": 0.3,
    "horse": 1.6, "sheep": 0.9, "cow": 1.4,
}
DEFAULT_HEIGHT_M = 0.3


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
        self.actual_width: int = config.frame_width
        self.actual_height: int = config.frame_height

    def start(self) -> None:
        # CAP_DSHOW avoids OpenCV silently ignoring resolution requests on Windows;
        # harmless no-op on other platforms since we fall back if it fails to open.
        self.cap = cv2.VideoCapture(self.cfg.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.cfg.camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self.cfg.camera_index}. "
                "Check that it's connected and not in use by another app."
            )

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_width = actual_w
        self.actual_height = actual_h
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
    """Detect -> diff against last frame -> announce newly appeared objects."""

    def __init__(self, config: Config):
        self.cfg = config
        self.camera = BlindAssistCamera(config)
        self.speech = SpeechEngine()
        print(f"Loading {config.model_path} on {config.device}...")
        self.model = YOLO(config.model_path)
        self.recent_passes: deque = deque(maxlen=config.vote_window)
        self.last_announced_at: dict = {}
        self.focal_length_px: float = 0.0  # set once the camera's actual resolution is known

    def _estimate_distance_m(self, label: str, box_height_px: float) -> float:
        """Rough monocular distance estimate: (real-world height * focal length) / pixel height.
        Accuracy depends on the FOV assumption and the per-class height table — treat this
        as 'closer/farther', not a precise measurement."""
        if box_height_px <= 0 or self.focal_length_px <= 0:
            return float("inf")
        real_height_m = KNOWN_HEIGHTS_M.get(label, DEFAULT_HEIGHT_M)
        return (real_height_m * self.focal_length_px) / box_height_px

    def _detect_objects(self, frame: np.ndarray) -> Set[str]:
        results = self.model(
            frame,
            conf=self.cfg.confidence,
            imgsz=self.cfg.inference_size,
            device=self.cfg.device,
            verbose=False,
        )
        names = self.model.names
        detected: Set[str] = set()
        for r in results:
            for box in r.boxes:
                label = names[int(box.cls[0])]
                if not self.cfg.enable_distance_filter:
                    detected.add(label)
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_height_px = y2 - y1
                distance_m = self._estimate_distance_m(label, box_height_px)
                if distance_m <= self.cfg.max_distance_meters:
                    detected.add(label)
        return detected

    def _stable_objects(self, current: Set[str]) -> Set[str]:
        """Only trust a label if it showed up in >= vote_threshold of the last
        vote_window detection passes. This absorbs single-frame flicker between
        similar-looking classes (e.g. bottle/banana/cup on an unfamiliar object)."""
        self.recent_passes.append(current)
        votes = Counter(label for pass_labels in self.recent_passes for label in pass_labels)
        return {label for label, count in votes.items() if count >= self.cfg.vote_threshold}

    def _announce_new_objects(self, stable_now: Set[str]) -> None:
        now = time.time()
        to_announce = [
            label for label in sorted(stable_now)
            if now - self.last_announced_at.get(label, 0.0) >= self.cfg.announce_cooldown
        ]
        if not to_announce:
            return
        suffix = " detected" if len(to_announce) == 1 else "s detected"
        message = ", ".join(to_announce) + suffix
        print(f"[SPEAK] {message}")
        self.speech.speak(message)
        for label in to_announce:
            self.last_announced_at[label] = now

    def run(self) -> None:
        self.camera.start()
        if self.cfg.enable_distance_filter:
            self.focal_length_px = self.camera.actual_width / (
                2 * math.tan(math.radians(self.cfg.camera_fov_degrees / 2))
            )
            print(f"Distance filter ON: max {self.cfg.max_distance_meters}m "
                  f"(focal length ~{self.focal_length_px:.0f}px, assumes "
                  f"{self.cfg.camera_fov_degrees}° FOV — tune camera_fov_degrees if "
                  f"distances look consistently too short/long)")
        print("BlindAssist running. Press 'q' in the video window to stop.")
        last_detection_time = 0.0
        try:
            while self.camera.is_running():
                frame = self.camera.get_frame()
                if frame is None:
                    continue

                now = time.time()
                if now - last_detection_time >= self.cfg.detection_interval:
                    current_objects = self._detect_objects(frame)
                    stable_now = self._stable_objects(current_objects)
                    self._announce_new_objects(stable_now)
                    last_detection_time = now

                cv2.imshow("BlindAssist", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            self.camera.stop()
            print("BlindAssist stopped.")


if __name__ == "__main__":
    app = BlindAssistApp(CFG)
    app.run()