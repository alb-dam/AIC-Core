"""Wrapper NDI Sender e gestore dual output (Native + AI)."""

import numpy as np
import cv2
import logging
from fractions import Fraction
from typing import Optional, Tuple

from cyndilib import Sender, VideoSendFrame, FourCC

logger = logging.getLogger(__name__)


class NDISender:
    """Singolo sender NDI: gestisce un canale video NDI con nome proprio."""

    def __init__(self, name: str, width: int, height: int, fps: int) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self._sender: Optional[Sender] = None
        self._vf: Optional[VideoSendFrame] = None
        self._frame_buffer: Optional[bytearray] = None
        self._frame_view: Optional[memoryview] = None

    def open(self) -> bool:
        """Inizializza e apre il sender NDI. Ritorna True se OK."""
        try:
            self._sender = Sender(self.name)
            
            self._vf = VideoSendFrame()
            self._vf.set_resolution(self.width, self.height)
            self._vf.set_frame_rate(Fraction(self.fps, 1))
            self._vf.set_fourcc(FourCC.BGRA)
            
            self._sender.set_video_frame(self._vf)
            
            frame_size = self._vf.get_data_size()
            self._frame_buffer = bytearray(frame_size)
            self._frame_view = memoryview(self._frame_buffer)
            
            self._sender.open()
            logger.info(f"NDI Sender '{self.name}' aperto ({self.width}x{self.height} @ {self.fps}fps)")
            return True
        except Exception as e:
            logger.error(f"Errore apertura NDI Sender '{self.name}': {e}")
            self._sender = None
            return False

    def send_frame(self, frame: np.ndarray) -> None:
        """Converte BGR→BGRA e invia il frame via NDI."""
        if self._sender is None or frame is None:
            return
        
        try:
            if frame.shape[2] == 3:
                bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            else:
                bgra = frame
            
            raw = bgra.tobytes()
            self._frame_buffer[:len(raw)] = raw
            
            self._sender.write_video_async(self._frame_view)
        except Exception as e:
            logger.error(f"Errore invio frame NDI '{self.name}': {e}")

    def close(self) -> None:
        """Chiude il sender NDI e libera le risorse."""
        if self._sender is not None:
            try:
                self._sender.close()
                logger.info(f"NDI Sender '{self.name}' chiuso.")
            except Exception as e:
                logger.error(f"Errore chiusura NDI Sender '{self.name}': {e}")
            finally:
                self._frame_view = None
                self._frame_buffer = None
                self._vf = None
                self._sender = None


class DualNDIOutput:
    """Gestore dei due output NDI contemporanei: Native (passthrough) e AI (elaborato)."""

    def __init__(self) -> None:
        self.ndi_ai: Optional[NDISender] = None
        self.ndi_native: Optional[NDISender] = None
        self._native_enabled: bool = False

    def initialize(self, ai_name: str, native_name: str, width: int, height: int, fps: int) -> None:
        """Inizializza entrambi i sender NDI."""
        self.close()

        self.ndi_ai = NDISender(ai_name, width, height, fps)
        if not self.ndi_ai.open():
            logger.error("Impossibile aprire il sender NDI AI.")
            self.ndi_ai = None

        self.ndi_native = NDISender(native_name, width, height, fps)
        if not self.ndi_native.open():
            logger.error("Impossibile aprire il sender NDI Native.")
            self.ndi_native = None

    def send_ai_frame(self, frame: np.ndarray) -> None:
        """Invia il frame elaborato dall'AI al sender NDI AI."""
        if self.ndi_ai is not None and frame is not None:
            self.ndi_ai.send_frame(frame)

    def send_native_frame(self, frame: np.ndarray) -> None:
        """Invia il frame raw/nativo al sender NDI Native (se abilitato)."""
        if self._native_enabled and self.ndi_native is not None and frame is not None:
            self.ndi_native.send_frame(frame)

    def set_native_enabled(self, enabled: bool) -> None:
        """Abilita o disabilita l'invio del frame nativo via NDI."""
        if self._native_enabled == enabled:
            return
        self._native_enabled = enabled
        state = "abilitato" if enabled else "disabilitato"
        logger.info(f"NDI Native output {state}.")

    @property
    def native_enabled(self) -> bool:
        return self._native_enabled

    @staticmethod
    def resize_and_pad(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Letterbox del frame alla dimensione target."""
        h, w = frame.shape[:2]
        tw, th = target_size

        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

        top = (th - nh) // 2
        bottom = th - nh - top
        left = (tw - nw) // 2
        right = tw - nw - left

        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])

    def close(self) -> None:
        """Chiude entrambi i sender NDI."""
        if self.ndi_ai is not None:
            self.ndi_ai.close()
            self.ndi_ai = None
        if self.ndi_native is not None:
            self.ndi_native.close()
            self.ndi_native = None
