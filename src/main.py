"""Entry point dell'applicazione AI-Cameraman (CLI Mode)."""

import logging
import signal
import time
import cv2

from src.utils.logger import setup_logging
from src.config.config import settings_run
from src.logic.pipeline import VideoPipeline

setup_logging()
logger = logging.getLogger(__name__)


def main_run() -> None:
    """Entry point CLI: istanzia, avvia e mantiene attiva l'applicazione."""
    settings = settings_run()
    
    # La VideoPipeline contiene tutta la logica di I/O, AI e Regia
    pipeline = VideoPipeline(settings_manager=settings)

    # Gestione shutdown pulito via SIGINT/SIGTERM
    def _shutdown(signum, frame):
        logger.info("Segnale di interruzione ricevuto. Chiusura in corso...")
        pipeline.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Avvio AI-Cameraman in modalità CLI (SRT → NDI)...")
    pipeline.start()

    debug_mode = settings.get("debug")
    if debug_mode:
        cv2.namedWindow("AI-Cameraman Debug", cv2.WINDOW_NORMAL)
        logger.info("Modalità Debug attivata. Finestra di anteprima creata.")

    # Mantieni il processo attivo
    try:
        while True:
            if debug_mode:
                dbg_frame = pipeline.get_debug_frame()
                if dbg_frame is not None:
                    cv2.imshow("AI-Cameraman Debug", dbg_frame)
                
                # Check for UI events and allow graceful close via 'q'
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    logger.info("Chiusura triggerata dalla finestra di debug.")
                    break
            else:
                time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        pipeline.stop()
        if debug_mode:
            cv2.destroyAllWindows()
        logger.info("AI-Cameraman terminato.")


if __name__ == "__main__":
    main_run()
