"""Modulo per il rendering di debug su schermo."""

import cv2
import numpy as np

from src.logic.detector import DetectionResult
from src.logic.director import CameraInstruction


class DebugRenderer:
    """Si occupa di disegnare i ritagli, i bounding box e i centri sul frame per il debug."""

    @staticmethod
    def draw_preview(
        frame: np.ndarray,
        det_out: DetectionResult,
        dir_out: CameraInstruction
    ) -> np.ndarray:
        """
        Disegna annotazioni di debug su una copia del frame originale.
        """
        debug_frame = frame.copy()

        # Disegna YOLO Detections (Players in Verde, Palla in Giallo)
        for p in det_out.raw_players:
            x1, y1, x2, y2 = p.box
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                debug_frame, f"P {p.conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        
        if det_out.raw_ball:
            x1, y1, x2, y2 = det_out.raw_ball.box
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                debug_frame, f"B {det_out.raw_ball.conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

        # Disegna Action Center (Rosso)
        if det_out.action_center:
            cx, cy = det_out.action_center
            cv2.circle(debug_frame, (int(cx), int(cy)), 8, (0, 0, 255), -1)
            cv2.putText(
                debug_frame, "Action Center", (int(cx) + 10, int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        # Disegna Smoothed Center (Blu)
        if dir_out.smoothed_center:
            scx, scy = dir_out.smoothed_center
            cv2.circle(debug_frame, (int(scx), int(scy)), 6, (255, 0, 0), -1)
            cv2.putText(
                debug_frame, "Smoothed Center", (int(scx) + 10, int(scy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )

        # Disegna Crop Box del Director (Azzurro/Blu)
        if dir_out.crop_box:
            cx1, cy1, cx2, cy2 = dir_out.crop_box
            cv2.rectangle(debug_frame, (cx1, cy1), (cx2, cy2), (255, 100, 0), 3)

        return debug_frame
