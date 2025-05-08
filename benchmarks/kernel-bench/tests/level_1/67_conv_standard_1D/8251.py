
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Non-float32 type not supported.
def test_non_float32_input():
    my_module = build_kernel()
    N = 2
    C_in = 3
    L_in = 16
    C_out = 4
    K = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1

    # Create input and weight tensors with double data type.
    x = torch.randn(N, C_in, L_in, device="cuda", dtype=torch.double)
    weight = torch.randn(C_out, C_in // groups, K, device="cuda", dtype=torch.double)
    bias = None

    with pytest.raises(RuntimeError, match="x must be a CUDA tensor"):
        # The host code in conv1d_forward_impl explicitly checks for at::kFloat;
        # hence, using double is expected to trigger a TORCH_CHECK error.
        _ = my_module.forward(x, weight, bias, stride, padding, dilation, groups)

# Issue 2: Invalid group configuration causing division-by-zero.
def test_invalid_group_configuration():
    my_module = build_kernel()
    N = 2
    C_in = 4   # For instance, 4 input channels.
    L_in = 16
    C_out = 4  # Number of output channels.
    K = 3
    stride = 1
    padding = 0
    dilation = 1
    # Improper groups such that groups > C_out (so that C_out / groups becomes zero)
    groups = 8  

    # Note: Although torch.nn.Conv1d requires C_in % groups == 0, our kernel implementation
    # does not check that C_out / groups is valid. This is expected to produce a runtime error,
    # for example via division by zero in computing group_idx.
    x = torch.randn(N, C_in, L_in, device="cuda", dtype=torch.float32)
    # According to PyTorch conv1d weight shape: [C_out, C_in/groups, K]
    weight = torch.randn(C_out, C_in // groups if C_in % groups == 0 else 1, K, device="cuda", dtype=torch.float32)
    bias = None

    with pytest.raises(RuntimeError):
        _ = my_module.forward(x, weight, bias, stride, padding, dilation, groups)

# Issue 3: Excessively large grid dimensions.
def test_excessive_grid_size():
    my_module = build_kernel()
    # We intentionally choose sizes that will yield a huge grid size.
    # Note: This test may trigger a CUDA launch error or result in allocation/read errors.
    # It is designed solely to stress the grid dimension computation.
    N = 100000      # large batch size
    C_in = 3
    L_in = 1024
    C_out = 64
    K = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1

    x = torch.randn(N, C_in, L_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_out, C_in // groups, K, device="cuda", dtype=torch.float32)
    bias = None

    # Expect that the huge grid dimension triggers a CUDA error.
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x, weight, bias, stride, padding, dilation, groups)
