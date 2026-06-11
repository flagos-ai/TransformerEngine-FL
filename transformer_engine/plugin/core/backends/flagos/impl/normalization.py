# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch
import flag_gems
from typing import Any, Dict, List, Optional, Tuple, Union


def layernorm_fwd_fl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    ln_out: Any,
    quantizer: Any,
    odtype: Any,
    sm_margin: int,
    zero_centered_gamma: bool,
) -> List[Any]:
    if zero_centered_gamma:
        # weight_adj = 1 + weight
        weight_adj = flag_gems.add(1, weight)
    else:
        weight_adj = weight

    y, mean, rstdevs = flag_gems.layer_norm(
        input,
        [input.shape[-1]],
        weight_adj,
        bias=bias,
        eps=eps,
    )

    if rstdevs.shape != input.shape[:-1]:
        rstdevs = rstdevs.view(input.shape[:-1])

    return y, mean, rstdevs


def layernorm_bwd_fl(
    dy: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int,
    zero_centered_gamma: bool,
) -> List[Any]:
    # When zero_centered_gamma is True, forward uses (1 + gamma) as weight
    # So backward needs to use (1 + gamma) for computing dx
    if zero_centered_gamma:
        gamma_adj = flag_gems.add(1, gamma)
    else:
        gamma_adj = gamma

    dummy_bias = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
    dx, dw, db = flag_gems.layer_norm_backward(
        dy, x, None, mu, rsigma, weight=gamma_adj, bias=dummy_bias, output_mask=[True, True, True]
    )

    return dx, dw, db
