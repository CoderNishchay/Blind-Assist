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
from deepface import DeepFace

logging.getLogger("ultralytics").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Config:
    model_path: str = "yolov8m.pt"
    confidence: float = 0.5
    inference_size: int = 640
    detection_interval: float = 1.0
    emotion_interval: float = 4.0
    vote_window: int = 3
    vote_threshold: int = 2
    camera_index: int = 0
    frame_width: int = 680
    frame_height: int = 480
    warmup_frames: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


class SpeechEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        threading.Thread(
            target=self._speak_sync,
            args=(text,),
            daemon=True
        ).start()

    def _speak_sync(self, text: str) -> None:
        with self._lock:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()


class BlindAssistCamera:

    def __init__(self, config: Config):
        self.cfg = config
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        self.cap = cv2.VideoCapture(
            self.cfg.camera_index,
            cv2.CAP_DSHOW
        )

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.cfg.camera_index)

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.cfg.frame_width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.cfg.frame_height
        )

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        actual_w = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        actual_h = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        print(f"Camera resolution: {actual_w}x{actual_h}")

        for _ in range(self.cfg.warmup_frames):
            self.cap.read()

    def is_running(self) -> bool:
        return (
            self.cap is not None
            and self.cap.isOpened()
        )

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

    def __init__(self, config: Config):

        self.cfg = config

        self.camera = BlindAssistCamera(config)
        self.speech = SpeechEngine()

        print(
            f"Loading {config.model_path} "
            f"on {config.device}..."
        )

        self.model = YOLO(config.model_path)

        self.recent_passes: deque = deque(
            maxlen=config.vote_window
        )

        self.currently_present: Set[str] = set()

        self._latest_frame: Optional[np.ndarray] = None

        self._latest_boxes: List[
            Tuple[int, int, int, int, str, float]
        ] = []

        self._frame_lock = threading.Lock()
        self._box_lock = threading.Lock()

        self._detector_stop = threading.Event()

        # Face announcement control
        self.last_face_names: Set[str] = set()
        self.last_face_time = 0.0

        # Emotion announcement control
        self.last_emotions: dict = {}
        self.last_emotion_analysis = 0.0

    def _detect_objects(
        self,
        frame: np.ndarray
    ) -> List[
        Tuple[int, int, int, int, str, float]
    ]:

        results = self.model(
            frame,
            conf=self.cfg.confidence,
            imgsz=self.cfg.inference_size,
            device=self.cfg.device,
            verbose=False,
        )

        names = self.model.names
        boxes = []

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                label = names[
                    int(box.cls[0])
                ]

                confidence = float(
                    box.conf[0]
                )

                boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        label,
                        confidence
                    )
                )

        return boxes

    def _detect_faces(
        self,
        frame: np.ndarray
    ) -> List[np.ndarray]:

        faces = []

        try:

            results = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False,
                align=True
            )

            for result in results:

                face = result.get("face")

                if face is None:
                    continue

                face = np.asarray(face)

                if face.size == 0:
                    continue

                # DeepFace may return RGB normalized image.
                # Convert to uint8 BGR for further processing.
                if face.dtype != np.uint8:
                    face = np.clip(
                        face * 255.0,
                        0,
                        255
                    ).astype(np.uint8)

                face = cv2.cvtColor(
                    face,
                    cv2.COLOR_RGB2BGR
                )

                faces.append(face)

        except Exception:
            pass

        return faces

    def _recognize_face(
        self,
        face: np.ndarray
    ) -> str:

        try:

            results = DeepFace.find(
                img_path=face,
                db_path="known_faces",
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=False,
                silent=True
            )

            if (
                len(results) > 0
                and len(results[0]) > 0
            ):

                identity = str(
                    results[0].iloc[0]["identity"]
                )

                if "Md Eliyas" in identity:
                    return "Md Eliyas"

        except Exception:
            pass

        return "Unknown person"

    def _detect_emotion(
        self,
        face: np.ndarray
    ) -> Optional[str]:

        try:

            result = DeepFace.analyze(
                img_path=face,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):

                if len(result) == 0:
                    return None

                result = result[0]

            emotion = result.get(
                "dominant_emotion"
            )

            if emotion:
                return str(
                    emotion
                ).capitalize()

        except Exception:
            pass

        return None

    def _announce_face(
        self,
        names: List[str]
    ) -> None:

        if not names:
            return

        current_time = time.time()

        current_names = set(names)

        if (
            current_names != self.last_face_names
            or current_time - self.last_face_time > 5
        ):

            for name in names:

                message = f"{name} detected"

                print(f"[FACE] {message}")

                self.speech.speak(message)

            self.last_face_names = current_names
            self.last_face_time = current_time

    def _announce_emotion(
        self,
        name: str,
        emotion: Optional[str]
    ) -> None:

        if emotion is None:
            return

        current_time = time.time()

        last_emotion = self.last_emotions.get(
            name,
            ""
        )

        if (
            emotion != last_emotion
            or current_time - self.last_emotion_analysis
            > self.cfg.emotion_interval
        ):

            message = (
                f"{name} appears to be {emotion}"
            )

            print(
                f"[EMOTION] {message}"
            )

            self.speech.speak(message)

            self.last_emotions[name] = emotion

    def _stable_objects(
        self,
        current: Set[str]
    ) -> Set[str]:

        self.recent_passes.append(current)

        votes = Counter(
            label
            for pass_labels in self.recent_passes
            for label in pass_labels
        )

        return {
            label
            for label, count in votes.items()
            if count >= self.cfg.vote_threshold
        }

    def _announce_new_objects(
        self,
        stable_now: Set[str]
    ) -> None:

        newly_arrived = (
            stable_now
            - self.currently_present
        )

        departed = (
            self.currently_present
            - stable_now
        )

        if newly_arrived:

            labels = sorted(newly_arrived)

            for label in labels:

                message = f"{label} detected"

                print(
                    f"[OBJECT] {message}"
                )

                self.speech.speak(message)

        if departed:

            labels = sorted(departed)

            for label in labels:

                message = f"{label} gone"

                print(
                    f"[OBJECT] {message}"
                )

                self.speech.speak(message)

        self.currently_present = stable_now

    @staticmethod
    def _draw_boxes(
        frame: np.ndarray,
        boxes: List[
            Tuple[int, int, int, int, str, float]
        ]
    ) -> np.ndarray:

        for (
            x1,
            y1,
            x2,
            y2,
            label,
            confidence
        ) in boxes:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            text = (
                f"{label} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

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

            if (
                now - last_detection_time
                < self.cfg.detection_interval
            ):
                time.sleep(0.01)
                continue

            # --------------------------------
            # YOLO OBJECT DETECTION
            # --------------------------------

            boxes = self._detect_objects(frame)

            current_objects = {
                label
                for *_, label, _ in boxes
            }

            stable_now = self._stable_objects(
                current_objects
            )

            self._announce_new_objects(
                stable_now
            )

            # --------------------------------
            # FACE DETECTION
            # --------------------------------

            faces = self._detect_faces(frame)

            detected_names = []

            for face in faces:

                # --------------------------------
                # FACE RECOGNITION
                # --------------------------------

                face_name = self._recognize_face(
                    face
                )

                detected_names.append(
                    face_name
                )

            # Announce known / unknown people
            self._announce_face(
                detected_names
            )

            # --------------------------------
            # EMOTION DETECTION
            # --------------------------------

            if (
                faces
                and now - self.last_emotion_analysis
                >= self.cfg.emotion_interval
            ):

                for index, face in enumerate(faces):

                    if index < len(detected_names):
                        face_name = detected_names[index]
                    else:
                        face_name = "Unknown person"

                    emotion = self._detect_emotion(
                        face
                    )

                    self._announce_emotion(
                        face_name,
                        emotion
                    )

                self.last_emotion_analysis = time.time()

            with self._box_lock:
                self._latest_boxes = boxes

            last_detection_time = time.time()

    def run(self) -> None:

        self.camera.start()

        print("BlindAssist running.")

        print(
            "Press 'q' in the camera window to stop."
        )

        detector_thread = threading.Thread(
            target=self._detection_worker,
            daemon=True
        )

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

                display_frame = self._draw_boxes(
                    frame.copy(),
                    boxes
                )

                cv2.imshow(
                    "BlindAssist",
                    display_frame
                )

                if (
                    cv2.waitKey(1) & 0xFF
                    == ord("q")
                ):
                    break

        except KeyboardInterrupt:

            print(
                "Interrupted by user."
            )

        finally:

            self._detector_stop.set()

            detector_thread.join(
                timeout=2.0
            )

            self.camera.stop()

            print(
                "BlindAssist stopped."
            )


if __name__ == "__main__":

    app = BlindAssistApp(CFG)

    app.run()