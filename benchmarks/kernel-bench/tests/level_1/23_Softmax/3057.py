
import math
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

# Issue 1: Kernel only supports float32.
def test_input_tensor_type():
    my_module = build_kernel()
    # Create a double tensor (float64) to trigger the TORCH_CHECK type error.
    x = torch.randn(16, 16384, dtype=torch.float64, device='cuda')
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor|Input tensor must be float32"):
        # This call should error out because of the dtype check
        my_module.forward(x)
    torch.cuda.synchronize()

# Issue 2: Kernel only supports 2D tensors.
def test_input_tensor_dim():
    my_module = build_kernel()
    # Create a 3D tensor to trigger the dimension check.
    x = torch.randn(4, 16, 16384, dtype=torch.float32, device='cuda')
    with pytest.raises(RuntimeError, match="Input tensor must be 2D"):
        my_module.forward(x)
    torch.cuda.synchronize()

# Issue 3: Entire row is -infinity which leads to division by zero (sum becomes 0) and NaNs in the output.
def test_all_minus_infinity():
    my_module = build_kernel()
    # Create a tensor where one entire row is -inf.
    batch_size = 4
    num_features = 100
    x = torch.randn(batch_size, num_features, dtype=torch.float32, device='cuda')
    # set one row to -inf
    x[2, :] = float("-inf")
    y = my_module.forward(x)
    torch.cuda.synchronize()
    # For the row that is -inf, softmax is undefined (0/0) and should produce NaNs.
    assert torch.isnan(y[2, :]).all(), "Expected NaNs for a row with all -infinity values."
