# transformer_engine_backend.py
import os

from transformer_engine.pytorch.cpp_extensions import (
    general_gemm,
)
from transformer_engine.pytorch.backend.gemm import (
    gems_general_gemm
)

class TransformerEngineBackend:
    _USE_TE_FL = bool(os.environ.get("USE_TRANSFORMER_ENGINE_FL", False))

    def use_te_fl(self) -> bool:
        return self._USE_TE_FL

    def gemm(self):
        return gems_general_gemm if self._USE_TE_FL else general_gemm

backend = TransformerEngineBackend()

