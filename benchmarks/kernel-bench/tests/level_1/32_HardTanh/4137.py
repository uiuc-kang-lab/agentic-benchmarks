
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu.
def build_kernel():
    # Ensure the local path is set correctly.
    src_file = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="hardtanh_kernel_module",
        sources=[src_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# 1. Test for incomplete processing with large input tensors.
def test_large_input():
    # Given the kernel uses a fixed maximum grid dimension of 65,535 and 256 threads per block,
    # the maximum number of threads is 256 * 65,535 = 16,777,  16,777,216. Any tensor with more elements
    # than that will be under-processed.
    threads = 256
    max_elems = 65_535 * threads
    # Create tensor with more than allowed number of elements.
    numel = max_elems + 1000
    x = torch.randn(numel, device="cuda", dtype=torch.float32)
    # Run the custom kernel.
    mod = build_kernel()
    out = mod.forward(x, -1.0, 1.0)
    # Compute reference using PyTorch's F.hardtanh.
    ref = F.hardtanh(x, min_val=-1.0, max_val=1.0)
    # Since the kernel did not process all elements, the results should differ.
    # We expect that not all elements are clipped correctly.
    # This test will fail if the kernel were fully correct.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "Kernel unexpectedly processed all elements despite grid size clamping, "
        "but it should have under-processed for large input tensors."
    )

# 2. Test for non-contiguous input tensor.
def test_non_contiguous():
    # Create a contiguous tensor then create a non-contiguous view (e.g. transpose of a 2D tensor).
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # transpose makes it non-contiguous
    assert not x_noncontig.is_contiguous(), "Test tensor must be non-contiguous."
    
    mod = build_kernel()
    try:
        out = mod.forward(x_noncontig, -1.0, 1.0)
    except Exception as e:
        pytest.skip("Kernel failed with non-contiguous input as expected: " + str(e))
    # Compare with reference computation.
    ref = F.hardtanh(x_noncontig, min_val=-1.0, max_val=1.0)
    # Likely the non-contiguous memory and reinterpret_casting might yield wrong results.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "Kernel produced correct output for a non-contiguous input. "
        "Expected misaligned accesses causing wrong results."
    )

# 3. Test for handling half (fp16) precision.
def test_half_precision():
    # Create a half precision tensor.
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    mod = build_kernel()
    out = mod.forward(x, -1.0, 1.0)
    ref = F.hardtanh(x, min_val=-1.0, max_val=1.0)
    # Since the kernel does not use vectorized loads for half precision,
    # performance may be impacted, but at least the results should be correct.
    # If results differ, then there is a correctness issue with half precision support.
    assert torch.allclose(out, ref, atol=1e-3), (
        "Kernel output for half precision does not match the reference. "
        "This highlights potential issues with non-vectorized handling of fp16."
    )

# 4. Test to ensure that passing a CPU tensor raises an exception.
def test_cpu_input():
    x = torch.randn(1024, dtype=torch.float32, device="cpu")
    mod = build_kernel()
    with pytest.raises(Exception) as excinfo:
        _ = mod.forward(x, -1.0, 1.0)
    assert "CUDA" in str(excinfo.value), (
        "Passing a non-CUDA tensor should raise an exception regarding CUDA device requirement."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
