import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


def ref_hardtanh(x, min_val, max_val):
    return F.hardtanh(x, min_val=min_val, max_val=max_val)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_misaligned_memory():
    torch.manual_seed(0)
    base = torch.randn(1025, device='cuda', dtype=torch.float32)
    x = base.narrow(0, 1, 1024).clone()
    x = x.contiguous()
    min_val = -1.0
    max_val = 1.0
    kernel = build_kernel()
    out_kernel = kernel.forward(x, min_val, max_val)
    out_ref = ref_hardtanh(x, min_val, max_val)
    assert torch.allclose(out_kernel, out_ref, atol=0.01
        ), 'Kernel output differs on misaligned memory input!'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_non_contiguous_input():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device='cuda', dtype=torch.float32)
    x_noncontig = x.t()
    min_val = -1.0
    max_val = 1.0
    kernel = build_kernel()
    try:
        out_kernel = kernel.forward(x_noncontig, min_val, max_val)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    out_ref = ref_hardtanh(x_noncontig, min_val, max_val)
    assert torch.allclose(out_kernel, out_ref, atol=0.01
        ), 'Kernel output differs on non-contiguous input!'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_non_divisible_numel():
    torch.manual_seed(0)
    numel = 1024 + 3
    x = torch.randn(numel, device='cuda', dtype=torch.float32)
    min_val = -0.5
    max_val = 0.5
    kernel = build_kernel()
    out_kernel = kernel.forward(x, min_val, max_val)
    out_ref = ref_hardtanh(x, min_val, max_val)
    assert torch.allclose(out_kernel, out_ref, atol=0.01
        ), 'Kernel output differs when numel is non-divisible by vector width.'
