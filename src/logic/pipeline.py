"""Gestisce il ciclo di vita e i 4 thread per l'I/O video e AI."""

import time
import threading
import logging
from typing import Optional
import numpy as np

from src.config.config import SettingsManager
from src.inout.input import SRTInput
from src.inout.output import DualNDIOutput
from src.logic.detector import Detector

from src.logic.queues import DropFrameQueue, WorkerThread
from src.logic.director import VirtualDirector
from src.logic.debug import DebugRenderer

logger = logging.getLogger(__name__)

class VideoPipeline:
    """Gestisce il ciclo di vita e i 4 thread per l'I/O video e AI."""

    def __init__(self, settings_manager: SettingsManager) -> None:
        self.settings = settings_manager

        # Componenti Core
        self.video_input = SRTInput()
        self.video_output = DualNDIOutput()
        self.detector = Detector(model_name=self.settings.get("yolo_model"))
        self.director = VirtualDirector()

        # Stato di runtime
        self.is_running: bool = False
        self.stop_event = threading.Event()
        self._empty_frames_count = 0
        self._last_reconnect_time: float = 0.0
        self._last_srt_shape: Optional[tuple] = None
        self._yolo_frame_counter: int = 0
        self.debug_frame: Optional[np.ndarray] = None
        
        # Code DropFrame
        self.q_capture_to_inference = DropFrameQueue(maxsize=2)
        self.q_inference_to_tracking = DropFrameQueue(maxsize=2)
        self.q_tracking_to_render = DropFrameQueue(maxsize=2)

        # Thread List
        self.threads: list[WorkerThread] = []

    def start(self) -> None:
        if self.is_running:
            self.stop()

        self.stop_event.clear()
        
        # Init Background Thread per non bloccare
        threading.Thread(target=self._async_initialize, daemon=True).start()

    def _async_initialize(self) -> None:
        port = int(self.settings.get("source_path"))
        try:
            logger.info(f"Inizializzazione SRT listener sulla porta {port}...")
            self.video_input.initialize(port=port)
            if self.stop_event.is_set():
                self.video_input.release()
                return
            logger.info(f"Sorgente SRT inizializzata.")
        except Exception as e:
            logger.warning(f"Sorgente SRT non avviata: {e}. Verranno fatti tentativi di riconnessione.")

        if self.stop_event.is_set():
            return

        out_fps = int(self.settings.get("output_fps"))
        out_w = int(self.settings.get("output_width"))
        out_h = int(self.settings.get("output_height"))

        self.video_output.initialize(
            ai_name=self.settings.get("ndi_ai_name"),
            native_name=self.settings.get("ndi_native_name"),
            width=out_w, height=out_h, fps=out_fps
        )
        self.video_output.set_native_enabled(True)
        
        self.q_capture_to_inference.clear()
        self.q_inference_to_tracking.clear()
        self.q_tracking_to_render.clear()
        self._empty_frames_count = 0
        self._yolo_frame_counter = 0

        self.threads = [
            WorkerThread("Capture", self._capture_loop, self.stop_event),
            WorkerThread("Inference", self._inference_loop, self.stop_event),
            WorkerThread("Tracking", self._tracking_loop, self.stop_event),
            WorkerThread("Render", self._render_loop, self.stop_event)
        ]

        for t in self.threads:
            t.start()

        self.is_running = True
        logger.info(f"Pipeline elaborazione avviata ({out_w}x{out_h} @ {out_fps}fps).")

    def stop(self) -> None:
        if not self.is_running and not self.stop_event.is_set():
            self.stop_event.set()
            return

        logger.info("Arresto pipeline in corso...")
        self.is_running = False
        self.stop_event.set()

        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2.0)

        self.threads.clear()

        try:
            self.video_input.release()
            self.video_output.close()
        except Exception as e:
            logger.error(f"Errore chiusura I/O: {e}")

        logger.info("Pipeline arrestata.")

    def get_debug_frame(self) -> Optional[np.ndarray]:
        """Restituisce l'ultimo frame con annotazioni di debug, se disponibile."""
        return self.debug_frame

    # ── Thread Loops ───────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        ret, frame = self.video_input.read_frame()
        if not ret:
            self._handle_empty_frame()
            return

        if self._last_srt_shape is None:
            self._last_srt_shape = frame.shape
        elif self._last_srt_shape != frame.shape:
            logger.warning(f"Rilevato cambio RTP ({self._last_srt_shape}->{frame.shape}). Forzo riavvio...")
            self._last_srt_shape = None
            self._last_reconnect_time = 0.0
            self._empty_frames_count = 1
            self._handle_empty_frame()
            return

        self._empty_frames_count = 0

        # Invio Native Passthrough 
        out_w = int(self.settings.get("output_width"))
        out_h = int(self.settings.get("output_height"))
        native_frame = self.video_output.resize_and_pad(frame, (out_w, out_h))
        self.video_output.send_native_frame(native_frame)
            
        self.q_capture_to_inference.put(frame)
        time.sleep(0.001)

    def _handle_empty_frame(self) -> None:
        self._empty_frames_count += 1
        
        if self._empty_frames_count > 0:
            now = time.time()
            if now - self._last_reconnect_time >= 0.5:
                self._last_reconnect_time = now
                if self.video_input.reconnect():
                    self._empty_frames_count = 0
                    self._last_srt_shape = None
            else:
                time.sleep(0.05)
            return
        time.sleep(0.03)

    def _inference_loop(self) -> None:
        frame = self.q_capture_to_inference.get(timeout=0.1)
        
        # Aggiornamento dinamico YOLO settings
        q_smooth = float(self.settings.get("kalman_q_smooth"))
        r_smooth = float(self.settings.get("kalman_r_smooth"))
        q_react = float(self.settings.get("kalman_q_reactive"))
        r_react = float(self.settings.get("kalman_r_reactive"))
        p = float(self.settings.get("kalman_preset_percent")) / 100.0
        q_std = q_smooth + (q_react - q_smooth) * p
        r_std = r_smooth + (r_react - r_smooth) * p
        roi_path = self.settings.get("roi")
        
        self.detector.set_config(q_std, r_std, int(self.settings.get("yolo_imgsz")), roi_path)

        # Skip frame logic
        interval = int(self.settings.get("yolo_inference_interval"))
        predict_only = (self._yolo_frame_counter % interval) != 0
        self._yolo_frame_counter += 1

        det_out = self.detector.process(frame, predict_only=predict_only)
        self.q_inference_to_tracking.put((frame, det_out))

    def _tracking_loop(self) -> None:
        frame, det_out = self.q_inference_to_tracking.get(timeout=0.1)
        
        # Aggiornamento dinamico Director settings
        self.director.set_config(
            fixed_zoom_percent=float(self.settings.get("fixed_zoom_percent")),
            dynamic_zoom_percent=float(self.settings.get("dynamic_zoom_percent")),
            max_spread=float(self.settings.get("director_max_spread")),
            dynamic_scale=float(self.settings.get("director_dynamic_scale")),
            zoom_smoothing=float(self.settings.get("director_zoom_smoothing")),
            zoom_deadzone=float(self.settings.get("director_zoom_deadzone")),
            pan_tilt_deadzone=float(self.settings.get("director_pan_tilt_deadzone")),
            pan_tilt_smoothing=float(self.settings.get("director_pan_tilt_smoothing"))
        )

        dir_out = self.director.process(frame, det_out.action_center, det_out.player_spread)
        self.q_tracking_to_render.put((dir_out.cropped_frame, frame, det_out, dir_out))

    def _render_loop(self) -> None:
        out_fps = float(self.settings.get("output_fps"))
        frame_duration = 1.0 / out_fps if out_fps > 0 else 0.033
        loop_start = time.time()
        
        obs_frame, original_frame, det_out, dir_out = self.q_tracking_to_render.get(timeout=0.1)
        
        out_w = int(self.settings.get("output_width"))
        out_h = int(self.settings.get("output_height"))
        
        ai_frame = self.video_output.resize_and_pad(obs_frame, (out_w, out_h))
        self.video_output.send_ai_frame(ai_frame)

        if self.settings.get("debug"):
            self.debug_frame = DebugRenderer.draw_preview(original_frame, det_out, dir_out)

        sleep_time = frame_duration - (time.time() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)
