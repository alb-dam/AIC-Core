"""Inferenza YOLO: riceve un frame, restituisce bounding box grezzi."""

import threading
import logging
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import numpy as np
import cv2

from src.utils.model import detect_device, load_optimized_model

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Bbox grezzo prodotto dal modello AI."""
    box: Tuple[int, int, int, int]
    center: Tuple[int, int]
    conf: float


class YoloDetector:
    """Inferenza YOLO: riceve un frame, restituisce bounding box grezzi."""

    PLAYER_CLASS_ID = 0
    BALL_CLASS_ID = 32
    MAX_BALL_SIZE_RATIO: float = 0.20
    MAX_BALL_ASPECT_RATIO: float = 3.0

    def __init__(self, model_name: str, ball_conf_thresh: float = 0.4) -> None:
        self.device, self.use_half = detect_device()
        self.model: Any = load_optimized_model(model_name, self.device, self.use_half)
        self.ball_conf_thresh: float = ball_conf_thresh
        self._inference_counter = 0
        self.roi_polygon: Optional[np.ndarray] = None

    def set_roi(self, roi_path: Optional[str]) -> None:
        if not roi_path or not os.path.exists(roi_path):
            self.roi_polygon = None
            if roi_path and not os.path.exists(roi_path):
                logger.warning(f"File ROI non trovato: {roi_path}")
            return
            
        try:
            with open(roi_path, 'r', encoding='utf-8') as f:
                points = json.load(f)
            self.roi_polygon = np.array(points, dtype=np.int32)
            logger.info(f"ROI caricata correttamente da {roi_path}")
        except Exception as e:
            logger.error(f"Errore durante il parsing del ROI in {roi_path}: {e}")
            self.roi_polygon = None

    def detect(self, frame: np.ndarray, imgsz: int = 640) -> Tuple[List[Detection], Optional[Detection]]:
        if self.model is None:
            return [], None

        try:
            results = self.model.predict(
                source=frame, imgsz=imgsz, verbose=False, half=self.use_half,
                device=self.device, classes=[self.PLAYER_CLASS_ID, self.BALL_CLASS_ID]
            )
        except Exception as e:
            self._free_memory()
            raise e
        finally:
            self._inference_counter += 1
            if self._inference_counter >= 1000:
                self._free_memory_async()
                self._inference_counter = 0

        raw_players: List[Detection] = []
        raw_ball: Optional[Detection] = None

        img_h, img_w = frame.shape[:2]
        max_ball_dim = min(img_w, img_h) * self.MAX_BALL_SIZE_RATIO

        for box in results[0].boxes:
            detection = self._parse_box(box)
            if detection is None:
                continue

            cls_id, entry = detection
            
            # Filtro ROI: ignora se fuori dall'area definita
            if self.roi_polygon is not None:
                try:
                    is_inside = cv2.pointPolygonTest(self.roi_polygon, (float(entry.center[0]), float(entry.center[1])), False) >= 0
                    if not is_inside:
                        continue
                except Exception as e:
                    logger.error(f"Errore durante pointPolygonTest: {e}")

            if cls_id == self.PLAYER_CLASS_ID:
                raw_players.append(entry)
            elif cls_id == self.BALL_CLASS_ID:
                if entry.conf < self.ball_conf_thresh:
                    continue

                bw = entry.box[2] - entry.box[0]
                bh = entry.box[3] - entry.box[1]
                aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)

                if bw > max_ball_dim or bh > max_ball_dim or aspect_ratio > self.MAX_BALL_ASPECT_RATIO:
                    continue

                if raw_ball is None or entry.conf > raw_ball.conf:
                    raw_ball = entry

        return raw_players, raw_ball

    def _free_memory_async(self) -> None:
        threading.Thread(target=self._free_memory, daemon=True).start()

    def _free_memory(self) -> None:
        try:
            import gc
            gc.collect(0)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _parse_box(box: Any) -> Optional[Tuple[int, Detection]]:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return cls_id, Detection(box=(x1, y1, x2, y2), center=(cx, cy), conf=conf)
