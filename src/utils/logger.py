"""Setup centralizzato del logging."""

import sys
import logging
import os
import glob
from datetime import datetime

def setup_logging(level: int = logging.INFO) -> None:
    """Configura il logging globale dell'applicazione."""
    # Assicurati che la directory logs esista
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Cancella i vecchi file di log
    for old_log in glob.glob(os.path.join(log_dir, '*.log')):
        try:
            os.remove(old_log)
        except OSError:
            pass
            
    # Crea un nuovo file con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'app_{timestamp}.log')
    
    # Rimuovi eventuali handler root esistenti
    for handler in list(logging.root.handlers):
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
