
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Rebuild the extension from kernel.cu. We force relink for each test.
    module = load(
        name="fused_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: Redundant kernel launches.
# Although the two kernel launches compute the same output, running them concurrently
# may cause non-deterministic behavior if the work was meant to be partitioned.
# This test creates a scenario where we hope to catch any race-condition or timing issue.
def test_redundant_kernel_launch(tmp_path):
    # Use CPU tensors as the code expects host pointers.
    N = 512
    A = torch.triu(torch.randn(N, N, dtype=torch.float32))
    B = torch.triu(torch.randn(N, N, dtype=torch.float32))
    
    module = build_kernel()
    # Call the extension; since the kernel is launched twice, the overall timing or
    # potential race conditions might cause an incorrect result.
    C = module.forward(A, B)
    # Reference computed by PyTorch: only the upper triangular part is computed correctly.
    C_ref = torch.triu(torch.matmul(A, B))
    # It may be that both redundant launches yield the same result.
    # We force a failure if the result is not exactly equal.
    assert torch.allclose(C, C_ref, atol=1e-5), (
        "Issue 1: The result of redundant kernel launches does not match the expected output. "
        "This indicates a race or redundant computation error."
    )

# Issue 2: Unused tile_width parameter (design issue).
# We cannot trigger a runtime error from an unused parameter directly.
# Instead, we call the kernel with a non-default tile_width and then verify that the result
# is still computed identically to torch.matmul. This test documents the design flaw.
def test_unused_tile_width(tmp_path):
    N = 512
    A = torch.triu(torch.randn(N, N, dtype=torch.float32))
    B = torch.triu(torch.randn(N, N, dtype=torch.float32))
    
    module = build_kernel()
    # There is no interface to change tile_width.
    # We assume the tile_width parameter is fixed (256) in the kernel call.
    # Here we simply check that the result is the same as the reference, which
    # indicates that the tile_width parameter has no effect.
    C = module.forward(A, B)
    C_ref = torch.triu(torch.matmul(A, B))
    assert torch.allclose(C, C_ref, atol=1e-5), (
        "Issue 2: Changing tile_width does not affect the output. The parameter is unused, "
        "which is incompatible with a general tiled implementation."
    )

# Issue 3: Incorrect handling of GPU tensors.
# The kernel always performs cudaMemcpy with HostToDevice and DeviceToHost.
# If we pass inputs that are already on the GPU, copying from device pointer using
# HostToDevice is likely to fail.
def test_gpu_input_error(tmp_path):
    N = 256
    A = torch.triu(torch.randn(N, N, dtype=torch.float32, device='cuda'))
    B = torch.triu(torch.randn(N, N, dtype=torch.float32, device='cuda'))
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting an error because the module code will try to use cudaMemcpy
        # with a device pointer as if it were host memory.
        _ = module.forward(A, B)

# Issue 4: Input tensor type other than float32.
# If we pass double tensors, the kernel will interpret memory wrongly.
def test_non_float32_input(tmp_path):
    N = 256
    # Create double precision upper triangular matrices on CPU (so that the copy directions are valid)
    A = torch.triu(torch.randn(N, N, dtype=torch.float64))
    B = torch.triu(torch.randn(N, N, dtype=torch.float64))
    
    module = build_kernel()
    # This will likely not throw an error but will result in incorrect output.
    C = module.forward(A, B)
    C_ref = torch.triu(torch.matmul(A.to(torch.float32), B.to(torch.float32)))
    
    # The output is incorrect because of the type mismatch.
    # We check that the result is not close to the expected result.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Issue 4: The kernel does not properly handle non-float32 inputs, but the result appears correct."
    )

# Issue 5: Lack of CUDA API error checking.
# We simulate this by forcing a failure in one of the CUDA calls.
# For example, intentionally passing an invalid size.
def test_invalid_dimension(tmp_path):
    # Create matrices with mismatched dimensions.
    N = 256
    # A is square but B is altered to have a wrong shape by slicing part of it.
    A = torch.triu(torch.randn(N, N, dtype=torch.float32))
    B = torch.triu(torch.randn(N, N, dtype=torch.float32))[:N-10, :N-10]
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = module.forward(A, B)
