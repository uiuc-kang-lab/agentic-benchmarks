
import os
import tempfile
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension from a given source file
def build_kernel(source_file):
    return load(
        name="test_module",
        sources=[source_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# Test case 1: Trigger an error by passing an unsupported dtype (float16).
# The kernel only supports float32 and float64.
def test_unsupported_dtype():
    # Build the kernel from the original kernel.cu file.
    cuda_module = build_kernel("kernel.cu")
    x = torch.randn(16, 16384, dtype=torch.float16, device="cuda")
    # Expect a RuntimeError because the TORCH_CHECK enforces the input type.
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, 1)

# Test case 2: Force a non-multiple-of-warp block size.
# We simulate this by temporarily modifying the kernel source on disk so that it forces a block size
# that is not a multiple of 32. Such a configuration can lead to shared-memory reduction errors.
def test_invalid_block_size():
    # Read original kernel code.
    with open("kernel.cu", "r") as f:
        original_code = f.read()
    
    # Create a modified version that forces an invalid block size.
    # Replace the optimal block size selection with a hard-coded value that is not a multiple of 32 (e.g. 50).
    # Note: This is only safe for the purpose of triggering the issue.
    modified_code = original_code.replace("int optimal_block_size = 256;", "int optimal_block_size = 50;")
    # For dimensions <=512 this replacement is unlikely to occur naturally, so we are forcing it.
    
    # Write the modified code to a temporary file.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False) as tmp:
        tmp.write(modified_code)
        tmp_file = tmp.name

    try:
        cuda_module = build_kernel(tmp_file)
        x = torch.randn(16, 512, dtype=torch.float32, device="cuda")
        # Compute reference LogSoftmax result.
        ref = torch.log_softmax(x, dim=1)
        out = cuda_module.forward(x, 1)
        torch.cuda.synchronize()
        # Since a block size that is not a multiple of warp size is used,
        # we expect the result to differ from the reference.
        assert not torch.allclose(out, ref, atol=1e-4), "Kernel unexpectedly produced correct results with invalid block size."
    finally:
        os.remove(tmp_file)

# Test case 3: Check that a misconfigured dimension (e.g. an out‐of‐range dim) in the forward call is caught.
# This indirectly points out that error checking after kernel launch is missing, as the check relies solely on host code.
def test_invalid_dim():
    cuda_module = build_kernel("kernel.cu")
    x = torch.randn(16, 16384, dtype=torch.float32, device="cuda")
    # Use an out-of-range value for the dimension.
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, 2)
