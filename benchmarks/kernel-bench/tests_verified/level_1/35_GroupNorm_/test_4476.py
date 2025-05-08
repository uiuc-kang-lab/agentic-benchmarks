import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

def test_constant_memory_overflow():
    kernel = build_kernel()
    batch_size = 2
    features = 2048
    num_groups = 8
    dim1 = 16
    dim2 = 16
    eps = 1e-05
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda', dtype=
        torch.float32)
    weight = torch.randn(features, device='cuda', dtype=torch.float32)
    bias = torch.randn(features, device='cuda', dtype=torch.float32)
    y_kernel = kernel.forward(x, weight, bias, num_groups, eps)
    group_norm_ref = nn.GroupNorm(num_groups=num_groups, num_channels=
        features, eps=eps).to(x.device)
    group_norm_ref.weight.data.copy_(weight)
    group_norm_ref.bias.data.copy_(bias)
    y_reference = group_norm_ref(x)
    assert torch.allclose(y_kernel, y_reference, atol=0.001
        ), 'Kernel output is unexpectedly close to the reference output despite constant memory overflow issue.'
