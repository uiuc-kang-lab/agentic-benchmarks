import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import threading
import pytest
import torch
from torch.utils.cpp_extension import load
import math


def run_pool3d(module, input, kernel_size, stride, padding, dilation,
    return_indices, ceil_mode):
    return module.forward(input, kernel_size, stride, padding, dilation,
        return_indices, ceil_mode)


def test_return_indices_api(tmp_path):
    module = build_kernel()
    batch_size, channels, D, H, W = 2, 3, 8, 8, 8
    input = torch.randn(batch_size, channels, D, H, W, device='cuda', dtype
        =torch.float32)
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = True
    ceil_mode = False
    out = run_pool3d(module, input, kernel_size, stride, padding, dilation,
        return_indices, ceil_mode)
    assert out.ndim == 6 and out.size(0
        ) == 2, 'API mismatch: Expected stacked output with first dim equal 2.'


def test_concurrent_invocations():
    module = build_kernel()
    batch_size, channels, D, H, W = 2, 3, 16, 16, 16
    input1 = torch.randn(batch_size, channels, D, H, W, device='cuda',
        dtype=torch.float32)
    input2 = torch.randn(batch_size, channels, D, H, W, device='cuda',
        dtype=torch.float32)
    params1 = 3, 2, 1, 1, False, False
    params2 = 3, 1, 0, 2, False, False
    results = [None, None]

    def worker(idx, inp, params):
        results[idx] = run_pool3d(module, inp, *params)
    t1 = threading.Thread(target=worker, args=(0, input1, params1))
    t2 = threading.Thread(target=worker, args=(1, input2, params2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert torch.allclose(results[0], results[1]
        ), 'Concurrent invocations might have caused constant memory data race leading to similar outputs.'

