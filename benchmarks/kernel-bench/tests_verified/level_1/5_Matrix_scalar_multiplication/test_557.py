import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load
import threading
import time


def test_concurrent_scalar_race():
    my_module = build_kernel()
    M, N = 512, 512
    s1 = 3.0
    s2 = 4.0
    A1 = torch.randn(M, N, device='cuda', dtype=torch.float32)
    A2 = torch.randn(M, N, device='cuda', dtype=torch.float32)
    results = {}

    def run_kernel(name, tensor, scalar):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            out = my_module.forward(tensor, scalar)
            stream.synchronize()
        results[name] = out
    thread1 = threading.Thread(target=run_kernel, args=('r1', A1, s1))
    thread2 = threading.Thread(target=run_kernel, args=('r2', A2, s2))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    C1_ref = A1 * s1
    C2_ref = A2 * s2
    assert torch.allclose(results['r1'], C1_ref, atol=0.01
        ), 'Concurrent execution: Kernel result with scalar s1 is incorrect.'
    assert torch.allclose(results['r2'], C2_ref, atol=0.01
        ), 'Concurrent execution: Kernel result with scalar s2 is incorrect.'
