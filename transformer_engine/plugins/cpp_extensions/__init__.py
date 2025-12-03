# Copyright (c) 2022-2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for c++ extensions"""
try:
    import flag_gems
    from .gemm import *
    from .rmsnorm import *
    HAVE_GEMS = True
except:
    HAVE_GEMS = False

from .fused_adam import *
from .multi_tensor_apply import *