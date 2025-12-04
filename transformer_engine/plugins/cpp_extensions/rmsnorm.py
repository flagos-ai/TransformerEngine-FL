# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..import_utils import safety_import, have_flag_gems

### RMSNORM
HAVE_FLAG_GEMS = have_flag_gems()

if HAVE_FLAG_GEMS:
    gems_rmsnorm_fwd = safety_import('transformer_engine.plugins.cpp_extensions.gems_rms_norm', 'rms_norm_forward')
    gems_rmsnorm_bwd = safety_import('transformer_engine.plugins.cpp_extensions.gems_rms_norm', 'rms_norm_backward')
else:
    gems_rmsnorm_fwd = None
    gems_rmsnorm_bwd = None

def rmsnorm_fwd_fl(
    input,
    weight,
    eps,
    ln_out,
    quantizer,
    odtype,
    sm_margin,
    zero_centered_gamma,
):
    assert HAVE_FLAG_GEMS, "GEMS is not installed"
    y, rstdevs = gems_rmsnorm_fwd(
        input,
        [input.shape[-1]],
        weight,
        eps,
    )
    return y, None, rstdevs


def rmsnorm_bwd_fl(
    dy,
    x,
    rsigma,
    gamma,
    sm_margin,
    zero_centered_gamma,
    eps,
):
    assert HAVE_FLAG_GEMS, "GEMS is not installed"
    dx, dw = gems_rmsnorm_bwd(
        dy,
        x,
        rsigma,
        [x.shape[-1]],
        gamma,
        eps,
    )
    return dx, dw
