"""Regia virtuale: compone CameraStrategy, filtri di smoothing e ritaglio frame."""

from typing import Tuple
from dataclasses import dataclass
import numpy as np

from src.utils.smoothing import ValueSmoother, PointSmoother, DeadzoneFilter, ScalarDeadzoneFilter
from src.utils.zoom import GeometryService, CameraStrategy

@dataclass
class CameraInstruction:
    """Istruzioni di regia generate dal Director."""
    cropped_frame: np.ndarray
    crop_box: Tuple[int, int, int, int]
    zoom_level: float
    smoothed_center: Tuple[int, int]


class VirtualDirector:
    """Regia virtuale: compone CameraStrategy, filtri di smoothing e ritaglio frame."""

    def __init__(self) -> None:
        self.camera_strategy = CameraStrategy()
        self.zoom_smoother = ValueSmoother(smoothing_factor=0.05, initial_value=1.0)
        self.zoom_deadzone = ScalarDeadzoneFilter(threshold=0.1)
        
        self.base_pan_tilt_deadzone: float = 0.05
        self.base_pan_tilt_smoothing: float = 0.03
        
        self.deadzone_filter = DeadzoneFilter(threshold_pct=self.base_pan_tilt_deadzone)
        self.pan_tilt_smoother = PointSmoother(smoothing_factor=self.base_pan_tilt_smoothing)

    def set_config(
        self, 
        fixed_zoom_percent: float, 
        dynamic_zoom_percent: float,
        max_spread: float = 1000.0,
        dynamic_scale: float = 1.0,
        zoom_smoothing: float = 0.05,
        zoom_deadzone: float = 0.1,
        pan_tilt_deadzone: float = 0.05,
        pan_tilt_smoothing: float = 0.03
    ) -> None:
        self.camera_strategy.set_config(
            fixed_zoom_percent, dynamic_zoom_percent,
            max_spread=max_spread, dynamic_scale=dynamic_scale
        )
        self.zoom_smoother.smoothing_factor = zoom_smoothing
        self.zoom_deadzone.threshold = zoom_deadzone
        self.base_pan_tilt_deadzone = pan_tilt_deadzone
        self.base_pan_tilt_smoothing = pan_tilt_smoothing
        self.deadzone_filter.threshold_pct = self.base_pan_tilt_deadzone
        self.pan_tilt_smoother.smoothing_factor = self.base_pan_tilt_smoothing

    def process(
        self, frame: np.ndarray,
        action_center: Tuple[int, int],
        player_spread: float
    ) -> CameraInstruction:
        target_pt = (float(action_center[0]), float(action_center[1]))
        h, w = frame.shape[:2]
        reference_length = float(np.hypot(w, h))

        raw_target_zoom = self.camera_strategy.compute_target_zoom(player_spread)
        stable_target_zoom = self.zoom_deadzone.filter(raw_target_zoom)
        current_zoom = self.zoom_smoother.smooth(stable_target_zoom)
        
        dynamic_pan_smoothing = self.base_pan_tilt_smoothing / current_zoom
        dynamic_pan_deadzone = self.base_pan_tilt_deadzone / current_zoom
        
        self.pan_tilt_smoother.smoothing_factor = dynamic_pan_smoothing
        self.deadzone_filter.threshold_pct = dynamic_pan_deadzone
        
        stable_target = self.deadzone_filter.filter(target_pt, reference_length)
        smoothed = self.pan_tilt_smoother.smooth(stable_target)

        center_int = (int(smoothed[0]), int(smoothed[1]))
        crop_box = GeometryService.calculate_crop_region(center_int, current_zoom, w, h)

        x1, y1, x2, y2 = crop_box
        cropped_frame = frame[y1:y2, x1:x2]

        return CameraInstruction(
            cropped_frame=cropped_frame,
            crop_box=crop_box,
            zoom_level=current_zoom,
            smoothed_center=center_int,
        )
