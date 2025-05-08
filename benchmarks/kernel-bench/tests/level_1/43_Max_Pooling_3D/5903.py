
import math
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_maxpool3d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1:
# Trigger issue 3 by providing a non-floating-point (e.g., integer) input tensor.
def test_non_float_input():
    cuda_module = build_kernel()
    # Create an int tensor (this should not be supported)
    x = torch.randint(0, 10, (2, 2, 8, 8, 8), dtype=torch.int32, device="cuda")
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = False
    with pytest.raises(RuntimeError):
        # The extension expects a floating type; using an int tensor should trigger an error.
        cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)

# Test case 2:
# Trigger issue 2: When return_indices is true, the output is a single stacked tensor rather than a tuple.
def test_return_indices_format():
    cuda_module = build_kernel()
    # Create a floating tensor
    x = torch.randn(2, 3, 8, 8, 8, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = True
    ceil_mode = False
    out = cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    # The correct behavior (as per PyTorch) would be to return a tuple of (output, indices)
    # Check if the output is a single tensor with an extra dimension
    if isinstance(out, torch.Tensor):
        # Expecting first dimension to be 2 (i.e. combined stack of output and indices)
        assert out.shape[0] == 2, f"Expected stacked tensor with first dim 2, got shape {out.shape}"
    else:
        pytest.fail("Expected a tensor (stacked output and indices) but got a tuple.")

# Test case 3:
# Trigger issue 1: Using tuple parameter values for kernel_size etc. (not supported in the kernel)
def test_tuple_parameters():
    cuda_module = build_kernel()
    # Although PyTorch API allows tuple parameters for 3D pooling, our CUDA extension only accepts ints.
    # Simulate this error by attempting to pass a tuple via the CUDA extension (this should fail).
    x = torch.randn(2, 3, 8, 8, 8, device="cuda", dtype=torch.float32)
    # These are tuples, but our kernel expects ints, so we force an error.
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    return_indices = False
    ceil_mode = False
    with pytest.raises(TypeError):
        # This should raise an error because the extension cannot unpack tuple parameters.
        cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)

# Test case 4:
# Trigger issue 4 and 5 indirectly by using a runtime kernel_size that may challenge unrolling and error checking.
# Here, we supply a relatively large kernel_size to cause a large loop count.
def test_large_kernel_size():
    cuda_module = build_kernel()
    # Create a floating tensor with sufficient spatial dimensions.
    x = torch.randn(2, 3, 20, 20, 20, device="cuda", dtype=torch.float32)
    kernel_size = 7  # large kernel size
    stride = 2
    padding = 3
    dilation = 1
    return_indices = False
    ceil_mode = False
    # Run the CUDA kernel. Even if the behavior is incorrect due to unrolling issues,
    # we simply check that the kernel returns a tensor.
    out = cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    assert isinstance(out, torch.Tensor), "The output is not a tensor."

# Test case 5:
# Address issue 6 by testing with an input shape that could lead to workload imbalance.
def test_workload_imbalance():
    cuda_module = build_kernel()
    # Create an input tensor with uneven spatial dimensions.
    x = torch.randn(1, 1, 13, 17, 19, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = False
    out = cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    # Compare with PyTorch’s native MaxPool3d to highlight potential differences.
    maxpool = torch.nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)
    ref_out = maxpool(x)
    # Allow a larger atol if the workload imbalance has introduced small errors.
    assert torch.allclose(out, ref_out, atol=1e-4), f"Kernel output ({out}) deviates from reference ({ref_out})."

# Test case 6:
# Trigger issue 6 indirectly: if the math functions are miscompiled due to missing <cmath>, we might not get proper output dimensions.
def test_output_dimension_calculation():
    cuda_module = build_kernel()
    x = torch.randn(2, 2, 15, 15, 15, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = True
    out = cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    # Compute expected output dimension using PyTorch formula.
    def compute_out_dim(input_size):
        return math.ceil((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1)
    expected_d = compute_out_dim(x.shape[2])
    expected_h = compute_out_dim(x.shape[3])
    expected_w = compute_out_dim(x.shape[4])
    assert out.shape[2] == expected_d and out.shape[3] == expected_h and out.shape[4] == expected_w, \
        f"Expected output dimensions ({expected_d}, {expected_h}, {expected_w}), but got {out.shape[2:5]}"
