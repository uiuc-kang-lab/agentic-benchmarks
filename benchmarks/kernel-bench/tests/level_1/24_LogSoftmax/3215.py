
import pytest
import torch
import math
from torch.utils.cpp_extension import load

# Function to build and load the CUDA kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_log_softmax",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to call the kernel forward
def cuda_log_softmax_forward(input, dim):
    module = build_kernel()
    return module.forward(input, dim)

# Issue 1: Test kernel with an empty reduction dimension.
def test_empty_reduction_dimension():
    # Create a tensor with non-zero batch dimension but empty softmax dimension.
    # For example, shape (batch_size, 0) when softmax is computed along dim=1.
    batch_size = 4
    dim_size = 0
    # Note: torch.log_softmax on an empty dimension returns an empty tensor.
    x = torch.randn(batch_size, dim_size, device="cuda", dtype=torch.float32)
    # The kernel will perform the permutation and then try to reduce over an empty dimension.
    # We expect the output to be an empty tensor too.
    out = cuda_log_softmax_forward(x, 1)
    # Verify that output shape is the same as input.
    assert out.shape == x.shape, f"Expected output shape {x.shape} but got {out.shape}."
    # Additionally verify that for an empty tensor, torch.log_softmax returns empty.
    out_ref = torch.log_softmax(x, dim=1)
    # If the kernel did a log(0) internally, it might yield -inf or NaN; so we want to catch that.
    if out.numel() == 0:
        pytest.skip("Empty reduction dimension: no numerical value to compare.")
    else:
        # This branch should not be reached.
        assert torch.allclose(out, out_ref, atol=1e-5), "Kernel output mismatch on empty reduction dimension."

# Issue 2: Test kernel with non-floating-point tensor.
def test_non_floating_tensor():
    # Create an integer tensor.
    batch_size = 4
    dim_size = 128
    x = torch.randint(0, 10, (batch_size, dim_size), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError) as excinfo:
        _ = cuda_log_softmax_forward(x, 1)
    # Check that the error message mentions the type requirement.
    assert "input must be a CUDA tensor" not in str(excinfo.value), "Unexpected error message."
    assert "input must be float32 or float64" in str(excinfo.value), "Incorrect error message for non-floating type."

# Issue 3: Test kernel with a very small reduction dimension.
def test_small_reduction_dimension():
    # Using a last dimension that is smaller than the typical warp size.
    batch_size = 8
    dim_size = 16  # much smaller than 32; many threads will be idle.
    x = torch.randn(batch_size, dim_size, device="cuda", dtype=torch.float32)
    out = cuda_log_softmax_forward(x, 1)
    out_ref = torch.log_softmax(x, dim=1)
    # Compare results; they should be close even for small reduction dimensions.
    assert torch.allclose(out, out_ref, atol=1e-5), f"Kernel output differs for small reduction dimension. Max difference: {(out - out_ref).abs().max().item()}"

if __name__ == "__main__":
    pytest.main([__file__])
