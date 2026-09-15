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
import queue as _queue
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

# ============================================================
# CUSTOM OBJECT CATEGORIES
# ============================================================

# 🟢 Daily / Common Objects
DAILY_OBJECTS = {
    "pen",
    "pencil",
    "book",
    "notebook",
    "remote",
    "keys",
    "wallet",
    "earphones",
    "charger",
    "mug",
    "glasses",
    "umbrella",
}

# 🟡 Important Objects
IMPORTANT_OBJECTS = {
    "medicine_box",
}

# 🔴 Dangerous Objects
DANGEROUS_OBJECTS = {
    "knife",
    "scissors",
    "blade",
    "cutter",
    "needle",
    "syringe",
    "broken_glass",
    "hammer",
}

@dataclass(frozen=True)
class Config:
    model_path: str = "yolov8n.pt"        # FIX 1: was "best.pt" — auto-downloads if missing
    confidence: float = 0.5
    inference_size: int = 640
    detection_interval: float = 0.8
    vote_window: int = 3
    vote_threshold: int = 1               # FIX 4: was 2 — objects were silently dropped
    camera_index: int = 0
    frame_width: int = 680
    frame_height: int = 480
    warmup_frames: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


# FIX 2: was creating a new pyttsx3 engine on every speak() call
# causing crashes and garbled speech. Now uses one engine + Queue.
class SpeechEngine:

    def __init__(self) -> None:
        self._queue: _queue.Queue = _queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text: str) -> None:
        self._queue.put(text)

    def _worker(self) -> None:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS ERROR] {e}")

    def stop(self) -> None:
        self._queue.put(None)


class BlindAssistCamera:
    """Wraps a local webcam via OpenCV."""

    def __init__(self, config: Config):
        self.cfg = config
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        # FIX 3: was only trying one index — now auto-tries 0 then 1
        opened = False
        for index in [self.cfg.camera_index, 1 - self.cfg.camera_index]:
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                opened = True
                break
            self.cap.release()

        if not opened:
            for index in [self.cfg.camera_index, 1 - self.cfg.camera_index]:
                self.cap = cv2.VideoCapture(index)
                if self.cap.isOpened():
                    opened = True
                    break
                self.cap.release()

        if not opened:
            raise RuntimeError(
                f"Could not open webcam at index {self.cfg.camera_index}. "
                "Check that it's connected and not in use by another app."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: requested {self.cfg.frame_width}x{self.cfg.frame_height}, "
              f"actual {actual_w}x{actual_h}")

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
        self.currently_present: Set[str] = set()

        self.object_directions = {}
        self.missed_detections = {}
        self.max_missed_detections = 3

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_boxes: List[
            Tuple[int, int, int, int, str, float, Optional[float], str]
        ] = []
        self._frame_lock = threading.Lock()
        self._box_lock = threading.Lock()
        self._detector_stop = threading.Event()

    def _get_direction(self, x1: int, x2: int, frame_width: int) -> str:
        object_center = (x1 + x2) // 2
        if object_center < frame_width * 0.33:
            return "left"
        elif object_center > frame_width * 0.66:
            return "right"
        else:
            return "ahead"

    def _detect_objects(
        self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str, float, Optional[float], str]]:

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
                box_width = x2 - x1
                distance = self._estimate_distance(label, box_width)
                direction = self._get_direction(x1, x2, frame.shape[1])
                boxes.append((x1, y1, x2, y2, label, conf, distance, direction))

        return boxes

    def _estimate_distance(self, label: str, box_width: int) -> Optional[float]:
        REAL_WIDTHS = {
            "person": 0.45,
            "cell phone": 0.075,
            "bottle": 0.07,
            "laptop": 0.32,
            "chair": 0.45,
            "backpack": 0.30,
            "cup": 0.08,
        }

        if box_width <= 0:
            return None

        real_width = REAL_WIDTHS.get(label.lower())

        if real_width is None:
            return None

        FOCAL_LENGTH = 700
        distance = (real_width * FOCAL_LENGTH) / box_width
        return round(distance, 1)

    def _stable_objects(self, current: Set[str]) -> Set[str]:
        self.recent_passes.append(current)
        votes = Counter(label for pass_labels in self.recent_passes for label in pass_labels)
        return {label for label, count in votes.items() if count >= self.cfg.vote_threshold}

    def _get_object_category(self, label: str) -> str:
        label = label.lower().strip()
        if label in DANGEROUS_OBJECTS:
            return "dangerous"
        if label in IMPORTANT_OBJECTS:
            return "important"
        if label in DAILY_OBJECTS:
            return "daily"
        return "normal"

    def _create_voice_message(self, label: str, direction: str) -> str:
        category = self._get_object_category(label)
        if direction == "ahead":
            location = "ahead of you"
        else:
            location = f"on your {direction}"
        if category == "dangerous":
            return f"Warning! {label} detected {location}. Please be careful."
        elif category == "important":
            return f"{label} detected {location}. Please take note."
        else:
            return f"{label} detected {location}."

    def _announce_new_objects(
        self,
        boxes: List[Tuple[int, int, int, int, str, float, Optional[float], str]]
    ) -> None:

        current_directions = {}
        for box in boxes:
            x1, y1, x2, y2, label, conf, distance, direction = box
            current_directions[label] = direction

        current_objects = set(current_directions.keys())

        newly_arrived = current_objects - self.currently_present

        if newly_arrived:
            dangerous = sorted(obj for obj in newly_arrived if obj in DANGEROUS_OBJECTS)
            important = sorted(obj for obj in newly_arrived if obj in IMPORTANT_OBJECTS)
            normal = sorted(
                obj for obj in newly_arrived
                if obj not in DANGEROUS_OBJECTS and obj not in IMPORTANT_OBJECTS
            )

            for label in dangerous:
                direction = current_directions[label]
                message = self._create_voice_message(label, direction)
                print(f"[DANGER] [SPEAK] {message}")
                self.speech.speak(message)

            for label in important:
                direction = current_directions[label]
                message = self._create_voice_message(label, direction)
                print(f"[IMPORTANT] [SPEAK] {message}")
                self.speech.speak(message)

            for label in normal:
                direction = current_directions[label]
                message = self._create_voice_message(label, direction)
                print(f"[SPEAK] {message}")
                self.speech.speak(message)

        for label in current_objects:
            if label in self.object_directions:
                old_direction = self.object_directions[label]
                new_direction = current_directions[label]
                if old_direction != new_direction:
                    if label in DANGEROUS_OBJECTS:
                        message = f"Warning! {label} is now {new_direction}. Please be careful."
                    else:
                        message = f"{label} is now {new_direction}."
                    print(f"[MOVE] [SPEAK] {message}")
                    self.speech.speak(message)

        # FIX 5: was `self.currently_present - current_objects` (a set)
        # iterated while calling discard() on it — mutation bug meant
        # missed_detections never reached threshold. Snapshot to list first.
        missing_objects = list(self.currently_present - current_objects)

        for label in missing_objects:
            self.missed_detections[label] = (
                self.missed_detections.get(label, 0) + 1
            )
            print(
                f"[MISS] {label}: "
                f"{self.missed_detections[label]}/"
                f"{self.max_missed_detections}"
            )
            if self.missed_detections[label] >= self.max_missed_detections:
                message = f"{label} gone."
                print(f"[GONE] [SPEAK] {message}")
                self.speech.speak(message)
                self.object_directions.pop(label, None)
                self.currently_present.discard(label)
                self.missed_detections.pop(label, None)

        for label in current_objects:
            self.missed_detections.pop(label, None)

        for label in current_objects:
            self.currently_present.add(label)
            self.object_directions[label] = current_directions[label]

    @staticmethod
    def _draw_boxes(
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int, str, float, Optional[float], str]]
    ) -> np.ndarray:

        for x1, y1, x2, y2, label, conf, distance, direction in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if distance is not None:
                text = f"{label} {conf:.2f} | ~{distance}m | {direction}"
            else:
                text = f"{label} {conf:.2f} | {direction}"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame

    def _detection_worker(self) -> None:
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
            current_objects = {box[4] for box in boxes}
            stable_now = self._stable_objects(current_objects)
            stable_boxes = [box for box in boxes if box[4] in stable_now]
            self._announce_new_objects(stable_boxes)

            with self._box_lock:
                self._latest_boxes = stable_boxes

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
            self.speech.stop()
            self.camera.stop()
            print("BlindAssist stopped.")


if __name__ == "__main__":
    app = BlindAssistApp(CFG)
    app.run()