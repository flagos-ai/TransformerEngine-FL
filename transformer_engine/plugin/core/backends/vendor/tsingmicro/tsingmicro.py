# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from ....ops import *


def _ensure_txda_available():
    global _txda_available
    try:
        import torch_txda

        return True
    except Exception as e:
        return False


def _check_txda_available() -> bool:
    if _ensure_txda_available():
        return True
    else:
        return False


class TXDABackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return _check_txda_available()

    def is_available(self) -> bool:
        return _check_txda_available()

    def get_flash_attention_class(self):
        raise NotImplementedError("get_flash_attention_class - not implemented in txda backend")
