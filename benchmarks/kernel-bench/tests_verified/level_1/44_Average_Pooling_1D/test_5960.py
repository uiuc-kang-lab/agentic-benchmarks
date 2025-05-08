import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load
import threading
import time
import cuda

def test_grid_dimension_limit():
    cuda_module = build_kernel()
    huge_in_channels = 70000
    batch_size = 1
    input_length = 128
    x = torch.randn(batch_size, huge_in_channels, input_length, device=
        'cuda', dtype=torch.float32)
    kernel_size = 3
    stride = 1
    padding = 1
    result_kernel = cuda_module.forward(x, kernel_size, stride, padding)
    expected_kernel = torch.nn.AvgPool1d(kernel_size, stride, padding)(x)
    diff = (result_kernel - expected_kernel).abs().max().item()
    assert diff < 0.01, f'Kernel produced incorrect results for large input size. Max difference: {diff}'

def test_concurrent_kernels():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    input_length = 32
    x = torch.randn(batch_size, in_channels, input_length, device='cuda',
        dtype=torch.float32)
    params1 = 3, 1, 1
    params2 = 5, 2, 2
    result = {}

    def launch_kernel(name, params, stream):
        with torch.cuda.stream(stream):
            out = cuda_module.forward(x, params[0], params[1], params[2])
            stream.synchronize()
            result[name] = out
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    t1 = threading.Thread(target=launch_kernel, args=('out1', params1, stream1)
        )
    t2 = threading.Thread(target=launch_kernel, args=('out2', params2, stream2)
        )
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    avg_pool1 = torch.nn.AvgPool1d(kernel_size=params1[0], stride=params1[1
        ], padding=params1[2])
    avg_pool2 = torch.nn.AvgPool1d(kernel_size=params2[0], stride=params2[1
        ], padding=params2[2])
    expected1 = avg_pool1(x)
    expected2 = avg_pool2(x)
    err1 = (result['out1'] - expected1).abs().max().item()
    err2 = (result['out2'] - expected2).abs().max().item()
    assert err1 < 1e-05 and err2 < 1e-05, f'Concurrent kernel launches produced incorrect results due to constant memory interference. Error for kernel1: {err1}, kernel2: {err2}'
