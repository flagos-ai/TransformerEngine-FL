"""Python-side device identity for the Enflame/GCU vendor backend."""

from __future__ import annotations

import torch


def apply_patch() -> None:
    """Point TE's device type at GCU when torch_gcu is usable (idempotent, best-effort).

    torch_gcu's transfer_to_gcu already maps torch.cuda.* onto torch.gcu.*, but
    tensors still report device type "gcu"; TE's device checks compare against
    transformer_engine.TE_DEVICE_TYPE, so it must say "gcu" like the other vendor
    backends set their own device type.
    """
    try:
        from .enflame import _check_enflame_available

        if not _check_enflame_available():
            return
    except Exception as e:  # pylint: disable=broad-except
        print(f"[TE-FL] Enflame backend not available: {e}")
        return

    # Do not import transformer_engine at module level: its __init__ imports this
    # module to run the patch.
    try:
        import transformer_engine

        transformer_engine.TE_DEVICE_TYPE = "gcu"
        transformer_engine.TE_PLATFORM = torch.gcu
    except Exception as e:  # pylint: disable=broad-except
        print(f"[TE-FL Enflame Patches] Error setting TE device type or platform: {e}")
