# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..import_utils import safety_import

### RMSNORM
try:
    gems_rmsnorm_fwd = safety_import('transformer_engine.plugins.cpp_extensions.gems_rms_norm', 'rms_norm_forward')
    gems_rmsnorm_bwd = safety_import('transformer_engine.plugins.cpp_extensions.gems_rms_norm', 'rms_norm_backward')
    HAVE_GEMS = True
except:
    gems_rmsnorm_fwd = None
    gems_rmsnorm_bwd = None
    HAVE_GEMS = False

def fl_rmsnorm_fwd(
    input,
    weight,
    eps,
    ln_out,
    quantizer,
    odtype,
    sm_margin,
    zero_centered_gamma,
):
    assert HAVE_GEMS, "GEMS is not installed"
    y, rstdevs = gems_rmsnorm_fwd(
        input,
        [input.shape[-1]],
        weight,
        eps,
    )
    return y, None, rstdevs


def fl_rmsnorm_bwd(
    dy,
    x,
    rsigma,
    gamma,
    sm_margin,
    zero_centered_gamma,
    eps,
):
    assert HAVE_GEMS, "GEMS is not installed"
    dx, dw = gems_rmsnorm_bwd(
        dy,
        x,
        rsigma,
        [x.shape[-1]],
        gamma,
        eps,
    )
    return dx, dw
