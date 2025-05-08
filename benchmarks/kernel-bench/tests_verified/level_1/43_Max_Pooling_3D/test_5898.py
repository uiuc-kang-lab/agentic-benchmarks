import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import torch
import pytest
from torch.utils.cpp_extension import load


def test_issue_shared_memory():
    input_tensor = torch.randn(64, 16, 32, 32, 32, device='cuda')
    model = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1)
    ref_out = model(input_tensor)
    kernel_module = build_kernel()
    kernel_out = kernel_module.forward(input_tensor, 3, 2, 1, 1, False, False)
    torch.cuda.synchronize()
    assert torch.allclose(kernel_out, ref_out, atol=0.01
        ), 'Kernel output does not match PyTorch MaxPool3d output. This may be due to the missing shared memory optimization.'


def test_issue_non_contiguous():
    input_tensor = torch.randn(16, 32, 64, 64, 64, device='cuda')
    non_contiguous = input_tensor.transpose(2, 3)
    kernel_module = build_kernel()
    try:
        out_noncontig = kernel_module.forward(non_contiguous, 3, 2, 1, 1, False,
            False)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous inputs.')
    torch.cuda.synchronize()
    out_contig = kernel_module.forward(non_contiguous.contiguous(), 3, 2, 1,
        1, False, False)
    torch.cuda.synchronize()
    assert torch.allclose(out_noncontig, out_contig, atol=0.01
        ), 'Kernel output differs between non-contiguous and contiguous inputs.'

def test_issue_return_indices_format():
    input_tensor = torch.randn(16, 32, 64, 64, 64, device='cuda')
    kernel_module = build_kernel()
    result = kernel_module.forward(input_tensor, 3, 2, 1, 1, True, False)
    torch.cuda.synchronize()
    assert result.dim() == 6 and result.size(0
        ) == 2, 'Return indices output format is not a stacked tensor with dim0==2. This deviates from expected behavior.'
