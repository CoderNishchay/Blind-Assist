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
    model_path: str = "best.pt"
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

# Remember the last known direction of each detected object
        self.object_directions = {}

        # Counts consecutive detection passes where each object is missing
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
        """Determine whether an object is on the left, ahead, or right."""

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

                # Object width in pixels
                box_width = x2 - x1

                # Estimate distance
                distance = self._estimate_distance(label, box_width)

                direction = self._get_direction(x1,x2,frame.shape[1])

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
        """Only trust a label if it showed up in >= vote_threshold of the last
        vote_window detection passes. This absorbs single-frame flicker between
        similar-looking classes (e.g. bottle/banana/cup on an unfamiliar object)."""
        self.recent_passes.append(current)
        votes = Counter(label for pass_labels in self.recent_passes for label in pass_labels)
        return {label for label, count in votes.items() if count >= self.cfg.vote_threshold}

    def _get_object_category(self, label: str) -> str:
        """Classify detected object by importance."""

        label = label.lower().strip()

        if label in DANGEROUS_OBJECTS:
            return "dangerous"

        if label in IMPORTANT_OBJECTS:
            return "important"

        if label in DAILY_OBJECTS:
            return "daily"

        return "normal"

    def _create_voice_message(self, label: str, direction: str) -> str:
        """Create voice message based on object category and direction."""

        category = self._get_object_category(label)

        if direction == "ahead":
            location = "ahead of you"
        else:
            location = f"on your {direction}"

        if category == "dangerous":
            return (
            f"Warning! {label} detected {location}. "
            "Please be careful."
        )

        elif category == "important":
            return (
            f"{label} detected {location}. "
            "Please take note."
        )

        else:
            return f"{label} detected {location}."
        
    def _announce_new_objects(
        self,
        boxes: List[
            Tuple[int, int, int, int, str, float, Optional[float], str]
        ]
    ) -> None:

        # ========================================================
        # CURRENT OBJECTS AND THEIR DIRECTIONS
        # ========================================================

        current_directions = {}

        for box in boxes:
            x1, y1, x2, y2, label, conf, distance, direction = box
            current_directions[label] = direction

        current_objects = set(current_directions.keys())

        # ========================================================
        # NEW OBJECTS
        # ========================================================

        newly_arrived = current_objects - self.currently_present

        if newly_arrived:

            # 🔴 Dangerous objects first
            dangerous = sorted(
                obj for obj in newly_arrived
                if obj in DANGEROUS_OBJECTS
            )

            # 🟡 Important objects
            important = sorted(
                obj for obj in newly_arrived
                if obj in IMPORTANT_OBJECTS
            )

            # 🟢 Daily / normal objects
            normal = sorted(
                obj for obj in newly_arrived
                if obj not in DANGEROUS_OBJECTS
                and obj not in IMPORTANT_OBJECTS
            )

            # 🔴 Dangerous
            for label in dangerous:

                direction = current_directions[label]

                message = self._create_voice_message(
                    label,
                    direction
                )

                print(f"[DANGER] [SPEAK] {message}")
                self.speech.speak(message)

            # 🟡 Important
            for label in important:

                direction = current_directions[label]

                message = self._create_voice_message(
                    label,
                    direction
                )

                print(f"[IMPORTANT] [SPEAK] {message}")
                self.speech.speak(message)

            # 🟢 Daily / Normal
            for label in normal:

                direction = current_directions[label]

                message = self._create_voice_message(
                    label,
                    direction
                )

                print(f"[SPEAK] {message}")
                self.speech.speak(message)

        # ========================================================
        # OBJECT CHANGED DIRECTION
        # ========================================================

        for label in current_objects:

            if label in self.object_directions:

                old_direction = self.object_directions[label]
                new_direction = current_directions[label]

                if old_direction != new_direction:

                    if label in DANGEROUS_OBJECTS:
                        message = (
                            f"Warning! {label} is now {new_direction}. "
                            "Please be careful."
                        )
                    else:
                        message = f"{label} is now {new_direction}."

                    print(f"[MOVE] [SPEAK] {message}")
                    self.speech.speak(message)

        # ========================================================
# OBJECT GONE
# ========================================================

        missing_objects = self.currently_present - current_objects

        for label in missing_objects:

    # Count consecutive missed detection passes
            self.missed_detections[label] = (
                self.missed_detections.get(label, 0) + 1
            )

            print(
                f"[MISS] {label}: "
                f"{self.missed_detections[label]}/"
                f"{self.max_missed_detections}"
            )

    # Announce "gone" only after 3 consecutive misses
            if self.missed_detections[label] >= self.max_missed_detections:

                message = f"{label} gone."

                print(f"[GONE] [SPEAK] {message}")
                self.speech.speak(message)

                self.object_directions.pop(label, None)

        # Remove from currently present after announcing gone
                self.currently_present.discard(label)

                self.missed_detections.pop(label, None)


# Reset miss counter if object is detected again
        for label in current_objects:
            self.missed_detections.pop(label, None)

        # ========================================================
        # UPDATE STATE
        # ========================================================

        self.currently_present = current_objects
        self.object_directions.update(current_directions)

    @staticmethod
    def _draw_boxes(
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int, str, float, Optional[float], str]]
    ) -> np.ndarray:

        for x1, y1, x2, y2, label, conf, distance, direction in boxes:
           

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            
            # Display distance and direction
            if distance is not None:
                text = f"{label} {conf:.2f} | ~{distance}m | {direction}"
            else:
                text = f"{label} {conf:.2f} | {direction}"

            # Get text size
            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                2
            )

            # Background for text
            cv2.rectangle(
                frame,
                (x1, y1 - th - 8),
                (x1 + tw + 4, y1),
                (0, 255, 0),
                -1
            )

            # Display text
            cv2.putText(
                frame,
                text,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

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
            current_objects = {box[4] for box in boxes}

            stable_now = self._stable_objects(current_objects)

            # Keep only stable detections
            stable_boxes = [
                box for box in boxes
                if box[4] in stable_now
                ]

            # Announce stable objects + direction
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
            self.camera.stop()
            print("BlindAssist stopped.")


if __name__ == "__main__":
    app = BlindAssistApp(CFG)
    app.run()