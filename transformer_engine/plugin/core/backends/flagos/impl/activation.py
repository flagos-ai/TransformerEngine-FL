import torch
from typing import Any
import flag_gems


def gelu_fl(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    return flag_gems.gelu(input, approximate="tanh")


def geglu_fl(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = input.chunk(2, dim=-1)
    return flag_gems.gelu(a, approximate="tanh") * b


def qgelu_fl(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    return input * flag_gems.sigmoid(1.702 * input)


def qgeglu_fl(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = input.chunk(2, dim=-1)
    return a * flag_gems.sigmoid(1.702 * a) * b


def relu_fl(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    return flag_gems.relu(input)


def reglu_fl(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = input.chunk(2, dim=-1)
    return flag_gems.relu(a) * b
