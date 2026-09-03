import os
from pathlib import Path
from typing import Tuple, Any
import torch
from nnsight import LanguageModel, VisionLanguageModel


def _load_env():
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v


_load_env()


SUPPORTED_MODELS = {
    "qwen-2.5-0.5b": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "default_ignition_layer": 12,
        "is_multimodal": False,
    },
    "llama-3.2-1b": {
        "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
        "default_ignition_layer": 10,
        "is_multimodal": False,
    },
    "llama-3.1-8b": {
        "repo_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "default_ignition_layer": 20,
        "is_multimodal": False,
    },
    "qwen-2.5-7b": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "default_ignition_layer": 18,
        "is_multimodal": False,
    },
    "gemma-4-e2b": {
        "repo_id": "google/gemma-4-E2B-it",
        "default_ignition_layer": 20,
        "is_multimodal": True,
    },
    "gemma-4-26b-a4b": {
        "repo_id": "google/gemma-4-26B-A4B-it",
        "default_ignition_layer": 20,
        "is_multimodal": True,
    },
}


def load_model(
    model_key: str = "llama-3.1-8b",
    custom_repo_id: str | None = None,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Any, int]:
    """
    Loads an nnsight LanguageModel or VisionLanguageModel with optimal ignition layer.
    """
    if custom_repo_id:
        repo_id = custom_repo_id
        is_multimodal = "gemma-4" in repo_id.lower()
        default_layer = 20
    else:
        info = SUPPORTED_MODELS.get(model_key.lower())
        if not info:
            raise ValueError(f"Unknown model_key: '{model_key}'. Choose from: {list(SUPPORTED_MODELS.keys())}")
        repo_id = info["repo_id"]
        is_multimodal = info["is_multimodal"]
        default_layer = info["default_ignition_layer"]

    print(f"[Model Loader] Initializing '{repo_id}' on device '{device}' with dtype={torch_dtype}...")
    hf_token = os.environ.get("HF_TOKEN")
    model_kwargs = {"device_map": device, "torch_dtype": torch_dtype}
    if hf_token:
        model_kwargs["token"] = hf_token

    if is_multimodal:
        model = VisionLanguageModel(repo_id, **model_kwargs)
    else:
        model = LanguageModel(repo_id, **model_kwargs)

    layers, _ = get_layers_and_head(model)
    total_layers = len(layers)
    ignition_layer = min(default_layer, total_layers - 2)

    print(f"[Model Loader] Loaded successfully. Total Layers: {total_layers}. Global Workspace Ignition Layer: {ignition_layer}")
    return model, ignition_layer


def get_layers_and_head(model: Any) -> Tuple[Any, Any]:
    """
    Extracts the sequential transformer layer list and lm_head across diverse architectures.
    """
    if hasattr(model.model, "layers"):
        return model.model.layers, model.lm_head
    elif hasattr(model.model, "language_model"):
        return model.model.language_model.model.layers, model.model.language_model.lm_head
    elif hasattr(model.model, "text_model"):
        return model.model.text_model.layers, model.lm_head
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h, model.lm_head
    raise AttributeError("Could not dynamically resolve transformer layers and lm_head.")
