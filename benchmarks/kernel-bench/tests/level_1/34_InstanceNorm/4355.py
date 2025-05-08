
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="instance_norm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

# Issue 1: Data type limitation
def test_input_dtype(kernel_module):
    # Create double precision (float64) input, weight, and bias.
    # The kernel interprets data as float32, so the output will be incorrect.
    N, C, H, W = 2, 2, 4, 4
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float64)
    weight = torch.randn(C, device="cuda", dtype=torch.float64)
    bias = torch.randn(C, device="cuda", dtype=torch.float64)
    y = kernel_module.forward(x, weight, bias, 1e-5)
    # Compute reference result with instance_norm in double precision.
    y_ref = torch.nn.functional.instance_norm(x, weight=weight, bias=bias, eps=1e-5)
    # The results should not be close due to the improper data type handling.
    assert not torch.allclose(y.double(), y_ref, atol=1e-5), \
        "Kernel incorrectly handled float64 input as float32 data!"

# Issue 2: blockDim.x assumption (non-multiple of warpSize)
def test_block_dim_assumption(kernel_module):
    # Although the forward() function fixes the launch to 256 threads (a multiple of 32),
    # simulate a scenario with a very small spatial domain that forces the reduction to operate on a non-full warp.
    # For example, if H*W is very small, some threads in the block will not have data.
    N, C, H, W = 1, 1, 8, 8  # HW=64, still less than 256; with a different blockDim configuration, reduction may be wrong.
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    y = kernel_module.forward(x, weight, bias, 1e-5)
    y_ref = torch.nn.functional.instance_norm(x, weight=weight, bias=bias, eps=1e-5)
    # In a wrong reduction, the computed mean/variance will be off.
    diff = (y - y_ref).abs().max().item()
    assert diff > 1e-3, \
        "Kernel reduction may be correct by accident but the test case did not trigger a reduction issue when blockDim.x is not a multiple of warpSize!"

# Issue 3: Weight and bias pointer logic limitation
def test_weight_bias_pointer_logic(kernel_module):
    # Provide weight but an empty bias tensor to simulate the case where only one parameter is defined.
    # The kernel's combined pointer check then prevents the affine transformation from running, 
    # leading to a result that does not match the expected behavior.
    N, C, H, W = 2, 3, 4, 4
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    # Simulate "undefined" bias with an empty tensor.
    bias = torch.empty(0, device="cuda", dtype=torch.float32)
    y = kernel_module.forward(x, weight, bias, 1e-5)
    # Without affine transformation, the correct behavior would be no scaling/shift.
    y_ref = torch.nn.functional.instance_norm(x, eps=1e-5)
    # The two outputs should differ because kernel incorrectly checks both parameters.
    with pytest.raises(AssertionError):
        assert torch.allclose(y, y_ref, atol=1e-5)

# Issue 4: Lack of kernel launch error checking
def test_kernel_launch_error_checking(kernel_module):
    # Trigger a launch error by providing tensors with mismatched expected dimensions.
    # For instance, if weight and bias have a different number of channels than x.
    x = torch.randn(2, 3, 4, 4, device="cuda", dtype=torch.float32)
    # Intentionally supply the wrong number of channels.
    weight = torch.randn(4, device="cuda", dtype=torch.float32)
    bias = torch.randn(4, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        y = kernel_module.forward(x, weight, bias, 1e-5)
        torch.cuda.synchronize()
