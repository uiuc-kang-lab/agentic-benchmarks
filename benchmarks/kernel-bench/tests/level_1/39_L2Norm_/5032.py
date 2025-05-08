
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel(extra_cuda_cflags=None):
    if extra_cuda_cflags is None:
        extra_cuda_cflags = ["-O3", "--use_fast_math"]
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Grid configuration fault
# Test with a case where C is such that blocksPerVector > expected number of Y blocks.
# For example, set C such that (C + threads - 1) / threads is > 1.
def test_incorrect_grid_configuration():
    # Using C value that forces multiple blocks per vector.
    batch_size = 4
    C = 300  # threads=256 => blocksPerVector = 2, but the grid.y is (2+1)/2 = 1, causing one block to process 300 elements instead of splitting properly.
    x = torch.randn(batch_size, C, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    # launch the kernel
    output = kernel.forward(x)
    # Compare with PyTorch's normalization along dim 1.
    norm = x.norm(p=2, dim=1, keepdim=True)
    expected = x / (norm + 1e-12)
    torch.cuda.synchronize()
    # If not all elements are processed correctly, the results will diverge.
    assert torch.allclose(output, expected, atol=1e-5), "Grid configuration error: Not all elements were processed. Check grid.y calculation."

# Issue 2: Tensor layout assumptions (fails for higher-dimensional or noncontiguous tensors)
def test_nonstandard_tensor_layout():
    # Create a 3D tensor, normalization is along dim 1.
    # In this layout, the assumption that outer_stride = input.stride(0) may break.
    x = torch.randn(2, 16, 8, device="cuda", dtype=torch.float32).transpose(1,2).contiguous()  # purposely muddle layout
    kernel = build_kernel()
    output = kernel.forward(x)
    norm = x.norm(p=2, dim=1, keepdim=True)
    expected = x / (norm + 1e-12)
    torch.cuda.synchronize()
    # This test should trigger a failure if the kernel does not properly handle non-standard memory layout.
    assert torch.allclose(output, expected, atol=1e-5), "Tensor layout error: The kernel did not correctly handle non-standard tensor layouts."

# Issue 3: Lack of half precision support.
def test_half_precision_input():
    batch_size = 8
    C = 256
    # Create a half precision tensor.
    x = torch.randn(batch_size, C, device="cuda", dtype=torch.float16)
    kernel = build_kernel()
    # Expect the kernel to fail or produce an error because half precision is not dispatched.
    with pytest.raises(RuntimeError):
        output = kernel.forward(x)
        torch.cuda.synchronize()

# Issue 4: No kernel launch error checking.
def test_kernel_launch_error_checking():
    # We simulate a kernel launch error by providing an input with invalid dimensions.
    # The kernel explicitly checks that input.dim()>=2, so passing a 1D tensor should trigger a TORCH_CHECK.
    x = torch.randn(100, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        output = kernel.forward(x)
        torch.cuda.synchronize()

# Issue 5: Shared memory assumption not robust for larger block sizes.
# We simulate this by launching the kernel with a non-standard threads per block.
def test_shared_memory_limit_assumption(monkeypatch):
    # Monkeypatch the build_kernel to compile with a higher thread count.
    # This test assumes that increasing the thread count (and hence warps) beyond 32 may cause wrong results.
    batch_size = 2
    C = 1024  # With 1024 threads, we have 1024/32 = 32 warps -- borderline; use one more thread to push it over.
    x = torch.randn(batch_size, C, device="cuda", dtype=torch.float32)
    
    # Adjust extra cuda flags to force a larger block size via macro definition (simulate expecting larger threads per block).
    # Note: This is a simulation. In a production scenario, kernel launch parameters would be passed in differently.
    kernel = build_kernel(extra_cuda_cflags=["-O3", "--use_fast_math", "-DTHREADS_PER_BLOCK=1056"])
    output = kernel.forward(x)
    norm = x.norm(p=2, dim=1, keepdim=True)
    expected = x / (norm + 1e-12)
    torch.cuda.synchronize()
    # If shared memory size assumption fails, the output will not match expected.
    assert torch.allclose(output, expected, atol=1e-5), "Shared memory error: Kernel produced unexpected result with high thread count."
    
if __name__ == "__main__":
    pytest.main([__file__])
