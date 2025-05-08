import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
import threading
from torch.utils.cpp_extension import load


def test_misaligned_input():
    my_module = build_kernel()
    base = torch.randn(1025, dtype=torch.float32, device='cuda')
    x = base.narrow(0, 1, 1024).contiguous()
    alpha = 1.0
    out = my_module.forward(x, alpha)
    out_ref = torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    assert torch.allclose(out, out_ref, atol=0.01
        ), 'Kernel unexpectedly produced correct result on misaligned tensor. Issue 1 is not triggered.'


def test_race_condition_constant_alpha():
    my_module = build_kernel()
    x = torch.randn(4096, dtype=torch.float32, device='cuda')

    def compute_expected(tensor, a):
        return torch.where(tensor > 0, tensor, a * (torch.exp(tensor) - 1))
    results = {}

    def worker(name, alpha_value):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            res = my_module.forward(x, alpha_value)
        torch.cuda.synchronize()
        results[name] = res
    t1 = threading.Thread(target=worker, args=('alpha1', 1.0))
    t2 = threading.Thread(target=worker, args=('alpha2', 2.0))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    out1_ref = compute_expected(x, 1.0)
    out2_ref = compute_expected(x, 2.0)
    err1 = (results['alpha1'] - out1_ref).abs().max().item()
    err2 = (results['alpha2'] - out2_ref).abs().max().item()
    assert err1 > 0.01 or err2 > 0.01, 'Kernel produced correct results for concurrent alpha values. Issue 2 is not triggered.'

def test_empty_input_tensor():
    my_module = build_kernel()
    x = torch.empty((0,), dtype=torch.float32, device='cuda')
    alpha = 1.0
    out = my_module.forward(x, alpha)
    assert out.numel(
        ) == 0, 'Kernel output should be an empty tensor but is not.'


if __name__ == '__main__':
    pytest.main([__file__])
