"""Filtri di smoothing e deadzone per valori scalari e punti 2D."""

from typing import Tuple, Optional

import numpy as np


class ValueSmoother:
    """Smoothing esponenziale per valori scalari."""

    def __init__(self, smoothing_factor: float = 0.10, initial_value: float = 1.0) -> None:
        self.smoothing_factor = smoothing_factor
        self.current_value = initial_value

    def smooth(self, target_value: float) -> float:
        self.current_value += (target_value - self.current_value) * self.smoothing_factor
        return self.current_value


class PointSmoother:
    """Smoothing esponenziale per punti 2D."""

    def __init__(self, smoothing_factor: float = 0.05) -> None:
        self.smoothing_factor = smoothing_factor
        self.current_point: Optional[Tuple[float, float]] = None

    def smooth(self, target_point: Tuple[float, float]) -> Tuple[float, float]:
        current = self.current_point
        if current is None:
            self.current_point = target_point
            return target_point

        cx, cy = current
        sx = cx + (target_point[0] - cx) * self.smoothing_factor
        sy = cy + (target_point[1] - cy) * self.smoothing_factor

        self.current_point = (sx, sy)
        return (sx, sy)


class DeadzoneFilter:
    """Filtro deadzone spaziale per punti 2D."""

    def __init__(self, threshold_pct: float = 0.05) -> None:
        self.threshold_pct = threshold_pct
        self.stable_point: Optional[Tuple[float, float]] = None

    def filter(self, target_point: Tuple[float, float], reference_length: float) -> Tuple[float, float]:
        stable = self.stable_point
        if stable is None:
            self.stable_point = target_point
            return target_point

        spx, spy = stable
        dx = target_point[0] - spx
        dy = target_point[1] - spy
        dist = float(np.hypot(dx, dy))

        threshold_px = self.threshold_pct * reference_length
        if dist <= threshold_px:
            return stable

        ratio = threshold_px / dist
        new_spx = target_point[0] - dx * ratio
        new_spy = target_point[1] - dy * ratio

        new_point = (new_spx, new_spy)
        self.stable_point = new_point
        return new_point


class ScalarDeadzoneFilter:
    """Filtro deadzone per valori scalari (es. zoom)."""

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold
        self.stable_value: Optional[float] = None

    def filter(self, target_value: float) -> float:
        stable = self.stable_value
        if stable is None:
            self.stable_value = target_value
            return target_value

        diff = abs(target_value - stable)

        if diff <= self.threshold:
            return stable

        if target_value > stable:
            new_val = target_value - self.threshold
        else:
            new_val = target_value + self.threshold

        self.stable_value = new_val
        return new_val
