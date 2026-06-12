import sys
import types

import torch


class _CheckpointBunch(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def _register_checkpoint_compat_modules():
    if "bunch" not in sys.modules:
        module = types.ModuleType("bunch")
        module.Bunch = _CheckpointBunch
        sys.modules["bunch"] = module


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_checkpoint(model, path, map_location="cpu", strict=True):
    try:
        checkpoint = torch.load(path, map_location=map_location)
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        _register_checkpoint_compat_modules()
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=strict)
    return checkpoint
