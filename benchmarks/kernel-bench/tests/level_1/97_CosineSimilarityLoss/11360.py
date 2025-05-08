
import os
import re
import torch
import pytest
from torch.utils.cpp_extension import load
import tempfile

# Utility function to build the CUDA extension from a given source string.
def build_kernel(source_str, extra_cuda_flags=None, extra_include_paths=None, module_name="test_module"):
    tmpdir = tempfile.TemporaryDirectory()
    source_file = os.path.join(tmpdir.name, "kernel.cu")
    with open(source_file, "w") as f:
        f.write(source_str)
    cuda_module = load(
        name=module_name,
        sources=[source_file],
        extra_cuda_cflags=extra_cuda_flags if extra_cuda_flags is not None else ["-O3", "--use_fast_math"],
        extra_include_paths=extra_include_paths if extra_include_paths is not None else [],
        verbose=False,
    )
    # Return both the module and the temporary directory (to keep it around)
    return cuda_module, tmpdir

# Read the original kernel.cu source from file.
# (For the purposes of this test, we assume that kernel.cu is in the same directory.)
with open("kernel.cu", "r") as f:
    original_source = f.read()

# Test case 1: Non-contiguous input.
#
# The kernel does not check for contiguity.
# Therefore, if predictions or targets are non-contiguous, the pointer arithmetic will be wrong.
# In this test we construct non-contiguous inputs and compare the kernel’s output with
# the expected loss computed using PyTorch’s built‐in cosine_similarity.
def test_non_contiguous_input():
    # Build the kernel using the original source
    cuda_module, tmpdir = build_kernel(original_source)
    # Create contiguous inputs first
    batch_size = 128
    D = 4096
    predictions = torch.randn(batch_size, D, device="cuda", dtype=torch.float32)
    targets = torch.randn(batch_size, D, device="cuda", dtype=torch.float32)
    # Now make them non-contiguous by transposing and then re-transposing (without calling contiguous())
    predictions_nc = predictions.t().t()  # This is a no-op logically but is not guaranteed to be contiguous
    targets_nc = targets.t().t()
    assert not predictions_nc.is_contiguous(), "Test input is unexpectedly contiguous"
    # Compute loss using our CUDA kernel extension
    loss_kernel = cuda_module.forward(predictions_nc, targets_nc)
    # Compute expected loss using PyTorch’s cosine similarity loss
    cosine_sim = torch.nn.functional.cosine_similarity(predictions_nc, targets_nc, dim=1)
    loss_reference = torch.mean(1 - cosine_sim)
    torch.cuda.synchronize()
    # Because the kernel assumes contiguous layout, the result should be noticeably different.
    # (If the kernel were robust, the two would be close.)
    assert not torch.allclose(loss_kernel, loss_reference, atol=1e-4), (
        f"Kernel loss did not differ from reference loss despite non-contiguous inputs!\n"
        f"Kernel loss: {loss_kernel.item()} vs reference loss: {loss_reference.item()}"
    )
    tmpdir.cleanup()

# Test case 2: Launch with a small block size.
#
# The kernel reduction code uses warp-level shuffle instructions that require at least 32 threads.
# In this test we simulate a faulty launch configuration by modifying the source code so that
# the host function uses a block size of 16 instead of 256.
#
# (Note: This test only makes sense if someone later calls the kernel with blockDim.x < 32.)
def test_small_block_size():
    # Modify the source to replace the host-side block_size from 256 to 16.
    modified_source = re.sub(r"const int block_size = 256;", "const int block_size = 16;", original_source)
    cuda_module, tmpdir = build_kernel(modified_source, module_name="test_module_small_block")
    # Use a simple input where the per-row vector has a moderate length (e.g. 4096)
    batch_size = 128
    D = 4096
    predictions = torch.randn(batch_size, D, device="cuda", dtype=torch.float32)
    targets = torch.randn(batch_size, D, device="cuda", dtype=torch.float32)
    # Compute loss using our CUDA kernel extension built with a small block size.
    loss_kernel = cuda_module.forward(predictions, targets)
    # Compute reference loss using PyTorch’s cosine similarity loss.
    cosine_sim = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
    loss_reference = torch.mean(1 - cosine_sim)
    torch.cuda.synchronize()
    # Because the reduction code assumes at least 32 threads, using 16 threads would give an incorrect result.
    # We test that the result is noticeably different from the expected loss.
    assert not torch.allclose(loss_kernel, loss_reference, atol=1e-4), (
        f"Kernel loss did not differ from reference loss despite a small block size!\n"
        f"Kernel loss: {loss_kernel.item()} vs reference loss: {loss_reference.item()}"
    )
    tmpdir.cleanup()

if __name__ == "__main__":
    pytest.main([__file__])
