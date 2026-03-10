# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

try:
    import torch_npu
except:
    pass
import torch.nn.functional as F

from ....ops import TEFLBackendBase, FP8TensorMeta

def _check_ascend_available() -> bool:
    import os
    try:
        import torch_npu
        return True
    except ImportError:
        print("[ASCEND] Disabled: import failed")
        return False


class AscendBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return _check_ascend_available()

    def __init__(self):
        pass

    def is_available(self) -> bool:
        return _check_ascend_available()

    def get_flash_attention_class(self):
        from .flash_attention import FlashAttentionASCEND
        return FlashAttentionASCEND

    def gelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return F.gelu(input)
