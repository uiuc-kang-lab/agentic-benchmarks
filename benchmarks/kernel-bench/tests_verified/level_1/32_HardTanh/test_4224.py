import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load
import time


def test_constant_memory_race_condition():
    my_module = build_kernel()
    x = torch.randn(16, 16384, dtype=torch.float32, device='cuda')
    min_val1, max_val1 = -1.0, 1.0
    min_val2, max_val2 = 0.5, 2.0
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream1):
        out1 = my_module.forward(x, min_val1, max_val1)
    time.sleep(0.001)
    with torch.cuda.stream(stream2):
        out2 = my_module.forward(x, min_val2, max_val2)
    torch.cuda.synchronize()
    expected1 = torch.clamp(x, min=min_val1, max=max_val1)
    expected2 = torch.clamp(x, min=min_val2, max=max_val2)
    err1 = (out1 - expected1).abs().max().item()
    err2 = (out2 - expected2).abs().max().item()
    assert err1 < 0.01, f'Kernel output for stream1 is incorrect: {err1}'
    assert err2 < 0.01, f'Kernel output for stream2 is incorrect: {err2}'
