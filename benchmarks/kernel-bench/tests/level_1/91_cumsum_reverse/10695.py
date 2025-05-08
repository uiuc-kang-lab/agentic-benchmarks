
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper to build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case to trigger Issue 1: Race condition from concurrent writes to __constant__ memory.
# We use an input tensor with an outer dimension > 1 (multiple rows) so that multiple blocks write concurrently.
def test_race_condition_in_constant_memory():
    # Create input shape where cumsum dimension is last and n <=1024.
    # Using shape (16, 512) ensures that 16 blocks are launched concurrently.
    torch.manual_seed(0)
    batch_size = 16
    n = 512
    # Build input tensor on CUDA.
    x = torch.randn(batch_size, n, device="cuda", dtype=torch.float32)
    
    # Expected result computed via PyTorch operations.
    x_expected = torch.cumsum(x.flip(-1), dim=-1).flip(-1)
    
    # Build the CUDA extension.
    cuda_module = build_kernel()
    # Call our kernel: the extension module exposes the "forward" function.
    x_result = cuda_module.forward(x, 1)
    torch.cuda.synchronize()

    # It is expected that due to race conditions across blocks writing into __constant__ memory,
    # the output does not match the expected result.
    assert not torch.allclose(x_result, x_expected, atol=1e-5), (
        "Race condition issue not triggered: kernel result unexpectedly matches expected output."
    )

# Test case to trigger Issue 2: Data type mismatch.
# Here we use an input tensor of type double. The kernel constant_data is of type float.
def test_dtype_mismatch_in_constant_memory():
    # Create input shape with last dimension and n<=1024.
    batch_size = 1
    n = 512
    # Using dtype=torch.double will force the kernel to use a type conversion when reading from constant memory.
    x = torch.randn(batch_size, n, device="cuda", dtype=torch.double)
    
    # Expected result computed via PyTorch operations.
    x_expected = torch.cumsum(x.flip(-1), dim=-1).flip(-1)
    
    cuda_module = build_kernel()
    x_result = cuda_module.forward(x, 1)
    torch.cuda.synchronize()

    # Because of the conversion from double to float in constant memory,
    # the output is expected to differ significantly from the reference.
    assert not torch.allclose(x_result, x_expected, atol=1e-5), (
        "Data type mismatch issue not triggered: kernel result unexpectedly matches expected output for double input."
    )
