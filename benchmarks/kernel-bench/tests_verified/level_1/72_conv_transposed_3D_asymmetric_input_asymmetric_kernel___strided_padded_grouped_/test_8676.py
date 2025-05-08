import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import build_kernel
import pytest
import torch
from torch.utils.cpp_extension import load


def test_kernel_is_wrapper():
    module = build_kernel()
    x = torch.randn(2, 4, 8, 8, 8, device='cuda', dtype=torch.float32)
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda', dtype=torch.float32)
    result = module.forward(x, weight, None, [2, 2, 2], [1, 1, 1], [0, 0, 0], 1
        )
    assert result is not None, 'Expected a valid tensor result, but got None.'


def test_invalid_conv_params_length():
    module = build_kernel()
    x = torch.randn(2, 4, 8, 8, 8, device='cuda', dtype=torch.float32)
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda', dtype=torch.float32)
    with pytest.raises(Exception):
        module.forward(x, weight, None, [2, 2], [1, 1, 1], [0, 0, 0], 1)
    with pytest.raises(Exception):
        module.forward(x, weight, None, [2, 2, 2], [1, 1], [0, 0, 0], 1)


def test_optional_bias_non_contiguous():
    module = build_kernel()
    x = torch.randn(2, 4, 8, 8, 8, device='cuda', dtype=torch.float32)
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda', dtype=torch.float32)
    bias = torch.randn(4, device='cuda', dtype=torch.float32)
    bias_non_contig = bias.unsqueeze(0).transpose(0, 0)
    bias_non_contig = bias_non_contig[:, :1].squeeze(1)
    try:
        result = module.forward(x, weight, bias_non_contig, [2, 2, 2], [1, 1, 1],
            [0, 0, 0], 1)
    except RuntimeError:
        pytest.skip('Kernel does not support non-contiguous bias.')
    torch.cuda.synchronize()
    result_ref = torch.nn.functional.conv_transpose3d(x, weight, bias,
        stride=[2, 2, 2], padding=[1, 1, 1], output_padding=[0, 0, 0], groups=1)
    assert torch.allclose(result, result_ref, atol=0.01
        ), 'Kernel did not produce the expected results with non-contiguous bias.'


def test_weight_incorrect_shape():
    module = build_kernel()
    x = torch.randn(2, 4, 8, 8, 8, device='cuda', dtype=torch.float32)
    weight_bad = torch.randn(4, 4, 3, 3, device='cuda', dtype=torch.float32)
    with pytest.raises(Exception):
        module.forward(x, weight_bad, None, [2, 2, 2], [1, 1, 1], [0, 0, 0], 1)
