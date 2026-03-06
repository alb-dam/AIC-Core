"""Gestione dell'input video esclusivamente tramite SRT."""

import cv2
import os
import logging
from typing import Tuple, Any, Optional

logger = logging.getLogger(__name__)


class SRTInput:
    """Gestione dell'input video SRT in ascolto su una porta specifica."""

    def __init__(self) -> None:
        """Inizializza la gestione dell'input senza avviare la cattura."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.port: Optional[int] = None

    def initialize(self, port: int = 9999) -> None:
        """Inizializza la sorgente video SRT in ascolto."""
        self.release()
        self.port = port

        srt_url = (
            f"srt://0.0.0.0:{port}"
            f"?mode=listener"
            f"&transtype=live"
            f"&latency=3000"
            f"&peerlatency=3000"
            f"&rcvbuf=16777216"
            f"&sndbuf=16777216"
            f"&pkt_size=1316"
            f"&tlpktdrop=0"
        )

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;2000000|listen_timeout;2000000|rw_timeout;2000000"

        logger.info(f"Avvio listener SRT su porta {port}: {srt_url}")
        cap = cv2.VideoCapture(srt_url, cv2.CAP_FFMPEG)

        if isinstance(cap, cv2.VideoCapture) and cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap = cap
        else:
            self.cap = cap

        if not self.cap.isOpened():
            raise RuntimeError(f"Impossibile avviare il listener SRT sulla porta {port}")

    def get_fps(self) -> float:
        """Ritorna gli FPS della sorgente. Default 30.0 se non disponibile."""
        if self.cap is None:
            return 30.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:
            return 30.0
        return float(fps)

    def read_frame(self) -> Tuple[bool, Any]:
        """Legge un singolo frame dalla sorgente inizializzata."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self) -> None:
        """Rilascia le risorse della sorgente video."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def reconnect(self) -> bool:
        """Tentativo di riconnessione alla porta configurata."""
        if self.port is None:
            logger.error("Impossibile riconnettere: porta non configurata.")
            return False
            
        logger.info(f"Tentativo di riconnessione SRT (porta {self.port})...")
        try:
            self.initialize(self.port)
            return True
        except Exception as e:
            logger.error(f"Riconnessione fallita: {e}")
            return False
