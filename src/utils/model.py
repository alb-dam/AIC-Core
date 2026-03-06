"""Caricamento e ottimizzazione dei modelli YOLO."""

import os
import sys
import logging
from typing import Tuple, Any

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    logger.warning("Modulo ultralytics (YOLO) non trovato. AI disabilitata.")
    YOLO = None  # type: ignore


def detect_device() -> Tuple[str, bool]:
    """Rileva il dispositivo di accelerazione disponibile.

    Returns:
        Tupla (device_string, use_half_precision).
    """
    device = "cpu"
    use_half = False
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda:0"
            use_half = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            use_half = True
    except ImportError:
        pass
    logger.info(f"YOLO configurato per usare device: {device}, precisione half: {use_half}")
    return device, use_half


def load_optimized_model(model_path: str, device: str, use_half: bool) -> Any:
    """Carica il modello YOLO con ottimizzazioni platform-specific.

    Su macOS prova CoreML (.mlpackage), altrimenti ONNX/TensorRT.

    Returns:
        Modello YOLO caricato, oppure None se fallisce.
    """
    if YOLO is None:
        return None

    base_name, _ = os.path.splitext(model_path)
    optimized_path = model_path

    if sys.platform == "darwin":
        mlpackage_path = base_name + ".mlpackage"
        if os.path.exists(mlpackage_path):
            optimized_path = mlpackage_path
        else:
            logger.info(f"Esportazione CoreML in corso per {model_path}...")
            try:
                temp_model = YOLO(model_path, task='detect')
                temp_model.export(format="coreml", nms=True)
                if os.path.exists(mlpackage_path):
                    optimized_path = mlpackage_path
            except Exception as e:
                logger.error(f"Errore esportazione CoreML: {e}")
    else:
        onnx_path = base_name + ".onnx"
        engine_path = base_name + ".engine"
        if os.path.exists(onnx_path):
            optimized_path = onnx_path
        elif os.path.exists(engine_path):
            optimized_path = engine_path
        else:
            logger.info(f"Esportazione ONNX in corso per {model_path}...")
            try:
                temp_model = YOLO(model_path, task='detect')
                temp_model.export(format="onnx", opset=12, half=use_half, device=device)
                if os.path.exists(onnx_path):
                    optimized_path = onnx_path
            except Exception as e:
                logger.error(f"Errore esportazione ONNX: {e}")

    try:
        model = YOLO(optimized_path, task='detect')
        if optimized_path.endswith('.pt') and device != "cpu":
            model.to(device)
        return model
    except Exception as e:
        logger.error(f"Errore caricamento modello {optimized_path}: {e}")
        return None
