"""Motore AI: combina YOLO, Kalman e calcolo del centro d'azione (ROI)."""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from src.inference.yolo import YoloDetector, Detection
from src.utils.kalman import KalmanTracker, TrackedObject

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Risultato completo di un ciclo di rilevamento + tracking."""
    players: List[TrackedObject] = field(default_factory=list)
    ball: Optional[TrackedObject] = None
    action_center: Tuple[int, int] = (0, 0)
    player_spread: float = 0.0
    raw_players: List[Detection] = field(default_factory=list)
    raw_ball: Optional[Detection] = None


class Detector:
    """Combina YOLO, Kalman e computo del baricentro (Action Center)."""

    def __init__(self, model_name: str, yolo_imgsz: int = 640) -> None:
        self.yolo = YoloDetector(model_name)
        self.yolo_imgsz = yolo_imgsz
        self.tracker = KalmanTracker()
        self.last_action_center: Optional[Tuple[int, int]] = None
        self.last_player_spread: float = 0.0
        self.roi_path: Optional[str] = None

    def set_config(self, q_std: float, r_std: float, yolo_imgsz: int = 640, roi_path: Optional[str] = None) -> None:
        self.tracker.set_config(q_std, r_std)
        self.yolo_imgsz = yolo_imgsz
        if roi_path != self.roi_path:
            self.roi_path = roi_path
            self.yolo.set_roi(self.roi_path)

    def process(self, frame: np.ndarray, predict_only: bool = False) -> DetectionResult:
        h, w = frame.shape[:2]
        frame_center = (w // 2, h // 2)

        if predict_only:
            raw_players, raw_ball = [], None
        else:
            raw_players, raw_ball = self.yolo.detect(frame, imgsz=self.yolo_imgsz)

        filtered_players, filtered_ball = self.tracker.update(raw_players, raw_ball, predict_only=predict_only)

        # Calcolo action center (peso 1.0 per giocatore, palla ignorata per stabilità)
        if filtered_players:
            xs = [p.center[0] for p in filtered_players]
            ys = [p.center[1] for p in filtered_players]
            computed_center = (int(np.mean(xs)), int(np.mean(ys)))
            self.last_action_center = computed_center
        else:
            computed_center = None

        action_center = self.last_action_center if self.last_action_center is not None else frame_center

        # Calcolo spread percentuale
        if filtered_players:
            max_xs, min_xs = max(xs), min(xs)
            max_ys, min_ys = max(ys), min(ys)
            spread_px = float(max(max_xs - min_xs, max_ys - min_ys))
            ref_len = float(np.hypot(w, h))
            self.last_player_spread = spread_px / ref_len if ref_len > 0 else 0.0

        return DetectionResult(
            players=filtered_players,
            ball=filtered_ball,
            action_center=action_center,
            player_spread=self.last_player_spread,
            raw_players=raw_players,
            raw_ball=raw_ball,
        )
