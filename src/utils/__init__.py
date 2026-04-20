"""Utility helpers shared across training, evaluation, and export."""

__all__ = ["get_best_available_device", "get_device_name"]


def __getattr__(name: str):
    if name in __all__:
        from src.utils.device import get_best_available_device, get_device_name

        return {
            "get_best_available_device": get_best_available_device,
            "get_device_name": get_device_name,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
