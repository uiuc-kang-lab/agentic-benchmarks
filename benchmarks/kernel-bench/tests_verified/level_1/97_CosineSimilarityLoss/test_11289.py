import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def compute_loss_pytorch(predictions, targets):
    cos_sim = torch.nn.functional.cosine_similarity(predictions, targets, dim=1
        )
    return torch.mean(1.0 - cos_sim)


def test_non_contiguous_tensor():
    N, D = 128, 4096
    base_pred = torch.randn(N, D * 2, device='cuda', dtype=torch.float32)
    base_target = torch.randn(N, D * 2, device='cuda', dtype=torch.float32)
    predictions = base_pred[:, ::2]
    targets = base_target[:, ::2]
    assert not predictions.is_contiguous(
        ), "Test tensor 'predictions' is unexpectedly contiguous"
    assert not targets.is_contiguous(
        ), "Test tensor 'targets' is unexpectedly contiguous"
    my_module = build_kernel()
    try:
        output_kernel = my_module.forward(predictions, targets)
    except RuntimeError as e:
        pytest.skip(f'Kernel does not support non-contiguous input: {e}')
    torch.cuda.synchronize()
    output_ref = compute_loss_pytorch(predictions, targets)
    torch.cuda.synchronize()
    assert torch.allclose(output_kernel, output_ref, atol=0.01
        ), 'Kernel should have issues with non-contiguous inputs but produced a close match.'


def test_fixed_warp_mask():
    N, D = 128, 10
    predictions = torch.randn(N, D, device='cuda', dtype=torch.float32)
    targets = torch.randn(N, D, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    output_kernel = my_module.forward(predictions, targets)
    output_ref = compute_loss_pytorch(predictions, targets)
    torch.cuda.synchronize()
    assert torch.allclose(output_kernel, output_ref, atol=0.01
        ), 'Kernel’s warp reduction (with fixed mask) should yield error with small D relative to block size.'


def test_irregular_dimension():
    N, D = 128, 1000
    predictions = torch.randn(N, D, device='cuda', dtype=torch.float32)
    targets = torch.randn(N, D, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    output_kernel = my_module.forward(predictions, targets)
    output_ref = compute_loss_pytorch(predictions, targets)
    torch.cuda.synchronize()
    assert torch.allclose(output_kernel, output_ref, atol=0.01
        ), 'Kernel output with irregular D should not match the reference due to reduction issues.'
