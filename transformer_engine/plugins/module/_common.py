# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..import_utils import safety_import

### RMSNORM
fl_rmsnorm_fwd = safety_import('transformer_engine.plugins.cpp_extensions', 'fl_rmsnorm_fwd')

def fl_apply_normalization(
    inputmat: torch.Tensor,
    ln_out: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: Union[torch.Tensor, None],
    eps: float,
    output_quantizer,
    output_dtype,
    normalization: str,
    fwd_ln_sm_margin: int,
    zero_centered_gamma: bool,
):
    normalization_func = fl_rmsnorm_fwd
    return normalization_func(
        inputmat,
        ln_weight,
        eps,
        ln_out,
        output_quantizer,
        output_dtype,
        fwd_ln_sm_margin,
        zero_centered_gamma,
    )
