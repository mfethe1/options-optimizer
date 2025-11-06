"""
Model Management Routes: Unified status across ML models

Provides GET /models/status aggregating training/persistence info for
Epidemic, TFT, GNN, Mamba, and PINN.
"""
from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()

# Optional imports of global instances from other route modules.
# If any import fails, we degrade gracefully and still return filesystem-based status.
try:
    from .gnn_routes import gnn_predictor  # type: ignore
except Exception as e:  # pragma: no cover
    logger.debug(f"GNN routes import failed in model_management: {e!r}")
    gnn_predictor = None  # type: ignore

try:
    from .advanced_forecast_routes import forecast_service  # type: ignore
except Exception as e:  # pragma: no cover
    logger.debug(f"Advanced forecast routes import failed: {e!r}")
    forecast_service = None  # type: ignore

try:
    from .mamba_routes import mamba_predictor  # type: ignore
except Exception as e:  # pragma: no cover
    logger.debug(f"Mamba routes import failed: {e!r}")
    mamba_predictor = None  # type: ignore

try:
    from .epidemic_volatility_routes import predictor as epidemic_predictor  # type: ignore
    from .epidemic_volatility_routes import trainer as epidemic_trainer  # type: ignore
except Exception as e:  # pragma: no cover
    logger.debug(f"Epidemic routes import failed: {e!r}")
    epidemic_predictor = None  # type: ignore
    epidemic_trainer = None  # type: ignore

try:
    from .pinn_routes import option_pricing_model  # type: ignore
except Exception as e:  # pragma: no cover
    logger.debug(f"PINN routes import failed: {e!r}")
    option_pricing_model = None  # type: ignore


def _file_info(path: str) -> Dict[str, Any]:
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    mtime = datetime.fromtimestamp(os.path.getmtime(path)).isoformat() if exists else None
    return {"path": path, "exists": exists, "size": size, "last_modified": mtime}


@router.get("/models/status")
async def get_models_status() -> Dict[str, Any]:
    """Aggregate model status including persistence and memory load flags."""
    # Expected weight/model paths
    paths = {
        "tft": os.path.join("models", "tft", "weights.weights.h5"),
        "gnn": os.path.join("models", "gnn", "weights.weights.h5"),
        "mamba": os.path.join("models", "mamba", "weights.weights.h5"),
        # Epidemic trainer saves a full Keras model per type
        "epidemic_seir": os.path.join("models", "epidemic", "SEIR_model.h5"),
        "epidemic_sir": os.path.join("models", "epidemic", "SIR_model.h5"),
        # PINN currently ephemeral; no persisted weights by design
    }

    # Compute loaded flags from globals when available
    tft_loaded = False
    try:
        if forecast_service and getattr(forecast_service.tft_predictor.tft, "is_trained", False):
            tft_loaded = True
    except Exception:
        pass

    gnn_loaded = False
    try:
        if gnn_predictor and getattr(gnn_predictor, "is_trained", False):
            gnn_loaded = True
    except Exception:
        pass

    mamba_loaded = False
    try:
        if mamba_predictor and getattr(mamba_predictor, "model", None) is not None:
            mamba_loaded = True
    except Exception:
        pass

    epidemic_loaded = bool(epidemic_predictor) or bool(epidemic_trainer)
    pinn_loaded = bool(option_pricing_model)

    # Assemble model entries
    models = [
        {
            "name": "TFT",
            "trained": os.path.exists(paths["tft"]) or tft_loaded,
            "weights": _file_info(paths["tft"]),
            "loaded_in_memory": tft_loaded,
        },
        {
            "name": "GNN",
            "trained": os.path.exists(paths["gnn"]) or gnn_loaded,
            "weights": _file_info(paths["gnn"]),
            "loaded_in_memory": gnn_loaded,
        },
        {
            "name": "Mamba",
            "trained": os.path.exists(paths["mamba"]) or mamba_loaded,
            "weights": _file_info(paths["mamba"]),
            "loaded_in_memory": mamba_loaded,
        },
        {
            "name": "Epidemic",
            "trained": os.path.exists(paths["epidemic_seir"]) or os.path.exists(paths["epidemic_sir"]) or epidemic_loaded,
            "weights": {
                "seir": _file_info(paths["epidemic_seir"]),
                "sir": _file_info(paths["epidemic_sir"]),
            },
            "loaded_in_memory": epidemic_loaded,
        },
        {
            "name": "PINN",
            "trained": pinn_loaded,  # currently ephemeral; no weights persisted
            "weights": None,
            "loaded_in_memory": pinn_loaded,
        },
    ]

    return {"models": models}

