"""Calcoli geometrici per crop region e strategia di zoom."""

from typing import Tuple

import numpy as np


class GeometryService:
    """Funzioni pure per calcoli geometrici e ritaglio."""

    @staticmethod
    def calculate_crop_region(
        center: Tuple[int, int], zoom: float, frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, int]:
        crop_w = int(frame_w / zoom)
        crop_h = int(frame_h / zoom)
        cx, cy = center

        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        return GeometryService.clamp_to_frame(x1, y1, x2, y2, crop_w, crop_h, frame_w, frame_h)

    @staticmethod
    def clamp_to_frame(
        x1: int, y1: int, x2: int, y2: int,
        crop_w: int, crop_h: int, frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, int]:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x1 + crop_w)
        y2 = min(frame_h, y1 + crop_h)

        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)

        return (x1, y1, x2, y2)


class CameraStrategy:
    """Strategia di zoom: computa lo zoom target basato su fixed + dynamic spread."""

    def __init__(self) -> None:
        self.fixed_zoom: float = 1.0
        self.dynamic_intensity: float = 0.0
        self._MAX_SPREAD: float = 1000.0
        self._DYNAMIC_SCALE: float = 1.0

    def set_config(
        self,
        fixed_zoom_percent: float,
        dynamic_zoom_percent: float,
        max_spread: float = 1000.0,
        dynamic_scale: float = 1.0
    ) -> None:
        self.fixed_zoom = 1.0 + (fixed_zoom_percent / 50.0)
        self.dynamic_intensity = dynamic_zoom_percent / 100.0
        self._MAX_SPREAD = max_spread
        self._DYNAMIC_SCALE = dynamic_scale

    def compute_target_zoom(self, player_spread: float) -> float:
        if self.dynamic_intensity == 0.0 or player_spread < 0:
            return self.fixed_zoom

        spread_norm = float(np.clip((self._MAX_SPREAD - player_spread) / self._MAX_SPREAD, 0.0, 1.0))
        dynamic_bonus = spread_norm * self.dynamic_intensity * self._DYNAMIC_SCALE
        return float(self.fixed_zoom + dynamic_bonus)
