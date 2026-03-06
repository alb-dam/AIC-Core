"""Filtro di Kalman 2D e tracker multi-oggetto."""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """Oggetto elaborato e stabilizzato dal tracking (Kalman)."""
    object_id: int
    center: Tuple[int, int]
    raw_box: Optional[Tuple[int, int, int, int]] = None


class SimpleKalman:
    """Filtro di Kalman 2D con preset di rumore configurabili."""

    def __init__(self, init_x: float, init_y: float, q_std: float = 20.0, r_std: float = 20.0) -> None:
        self.dt: float = 1.0 / 30.0
        self.x: np.ndarray = np.array([[init_x], [init_y], [0.0], [0.0]])

        self.F: np.ndarray = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.H: np.ndarray = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q: np.ndarray = np.eye(4)
        self.R: np.ndarray = np.eye(2)
        self.P: np.ndarray = np.eye(4) * 10.0
        self.time_since_update: int = 0
        self.set_config(q_std, r_std)

    def set_config(self, q_std: float, r_std: float) -> None:
        self.Q = np.eye(4) * q_std
        self.R = np.eye(2) * r_std

    def predict(self) -> Tuple[float, float]:
        self.time_since_update += 1
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, z_x: float, z_y: float) -> Tuple[float, float]:
        self.time_since_update = 0
        Z = np.array([[z_x], [z_y]])
        y = Z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        if abs(det) < 1e-12:
            det = 1e-12
        S_inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det

        K = np.dot(np.dot(self.P, self.H.T), S_inv)
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)
        return float(self.x[0, 0]), float(self.x[1, 0])


class KalmanTracker:
    """Classifica e tracca le bbox grezze usando multiple istanze Kalman."""

    MAX_MATCH_DISTANCE = 150.0

    def __init__(self, q_std: float = 20.0, r_std: float = 20.0) -> None:
        self.q_std: float = q_std
        self.r_std: float = r_std
        self.players_kalman: Dict[int, SimpleKalman] = {}
        self.ball_kalman: Optional[SimpleKalman] = None
        self.next_player_id: int = 0

    def set_config(self, q_std: float, r_std: float) -> None:
        self.q_std = q_std
        self.r_std = r_std
        if self.ball_kalman:
            self.ball_kalman.set_config(q_std, r_std)
        for k in self.players_kalman.values():
            k.set_config(q_std, r_std)

    def update(
        self,
        raw_players: list,
        raw_ball,
        predict_only: bool = False
    ) -> Tuple[List[TrackedObject], Optional[TrackedObject]]:
        if self.ball_kalman:
            self.ball_kalman.predict()
        for k in self.players_kalman.values():
            k.predict()

        if predict_only:
            filtered_ball = None
            if self.ball_kalman:
                filtered_ball = TrackedObject(
                    object_id=-1,
                    center=(int(self.ball_kalman.x[0, 0]), int(self.ball_kalman.x[1, 0])),
                    raw_box=None
                )
            filtered_players = [
                TrackedObject(
                    object_id=pid,
                    center=(int(k.x[0, 0]), int(k.x[1, 0])),
                    raw_box=None
                )
                for pid, k in self.players_kalman.items()
            ]
            return filtered_players, filtered_ball

        # Aggiorna palla
        filtered_ball = None
        if raw_ball:
            cx, cy = raw_ball.center
            if self.ball_kalman is None:
                self.ball_kalman = SimpleKalman(cx, cy, q_std=self.q_std, r_std=self.r_std)
            up_x, up_y = self.ball_kalman.update(cx, cy)
            filtered_ball = TrackedObject(object_id=-1, center=(int(up_x), int(up_y)), raw_box=raw_ball.box)
        elif self.ball_kalman:
            filtered_ball = TrackedObject(
                object_id=-1,
                center=(int(self.ball_kalman.x[0, 0]), int(self.ball_kalman.x[1, 0])),
                raw_box=None
            )

        # Aggiorna giocatori
        filtered_players: List[TrackedObject] = []
        new_kalman: Dict[int, SimpleKalman] = {}

        for rp in raw_players:
            cx, cy = rp.center
            best_id, best_dist = -1, self.MAX_MATCH_DISTANCE
            for pid, k in self.players_kalman.items():
                dist = float(np.hypot(cx - float(k.x[0, 0]), cy - float(k.x[1, 0])))
                if dist < best_dist:
                    best_id, best_dist = pid, dist

            if best_id != -1:
                k = self.players_kalman.pop(best_id)
                up_x, up_y = k.update(cx, cy)
                new_kalman[best_id] = k
                filtered_players.append(
                    TrackedObject(object_id=best_id, center=(int(up_x), int(up_y)), raw_box=rp.box)
                )
            else:
                k = SimpleKalman(cx, cy, q_std=self.q_std, r_std=self.r_std)
                new_kalman[self.next_player_id] = k
                filtered_players.append(
                    TrackedObject(object_id=self.next_player_id, center=(cx, cy), raw_box=rp.box)
                )
                self.next_player_id += 1

        for pid, k in self.players_kalman.items():
            if k.time_since_update < 30:
                new_kalman[pid] = k

        self.players_kalman = new_kalman
        return filtered_players, filtered_ball
