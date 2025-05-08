
import torch
import pytest
from torch.utils.cpp_extension import load

# Function to build the extension module from kernel.cu
def build_kernel():
    return load(
        name="l2norm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Reference implementation: simple L2 normalization over dim=1, matching the model forward pass.
def reference_l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)

# Issue 1: Test misaligned pointer assumption.
# We force misalignment by creating a storage buffer larger than needed and using as_strided with a nonzero storage_offset.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_misaligned_input():
    # Create a contiguous tensor, then use as_strided to shift the starting offset by 1 element.
    batch_size = 16
    dim = 16384
    # Allocate extra element so we can offset the storage pointer.
    storage = torch.empty(batch_size * dim + 1, device='cuda', dtype=torch.float32)
    # Create tensor view with a storage offset of 1, preserving shape and normal strides.
    x = torch.as_strided(storage[1:], size=(batch_size, dim), stride=(dim, 1))
    
    # x is contiguous as a view but its underlying data is misaligned.
    kernel_module = build_kernel()
    y_kernel = kernel_module.forward(x)
    y_ref = reference_l2norm(x)
    # The misaligned accesses may lead to wrong results.
    err = (y_kernel - y_ref).abs().max().item()
    # We expect the error to be significant because the kernel is not safe for misaligned memory.
    assert err > 1e-3, f"Expected significant error due to misaligned memory access, got error {err}"

# Issue 2: Test unsupported data type (half precision).
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_half_precision():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float16)
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because half precision is not dispatched.
        _ = kernel_module.forward(x)

# Issue 3: Test input with higher dimensions.
# The kernel computes the base offset using outer_stride = input.stride(0), which is incorrect when input is not strictly 2D.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_higher_dimensional_input():
    # Create a 3D tensor of shape (B, C, D) and normalize along dim 1.
    B, C, D = 4, 16384, 2
    # Create a contiguous 3D tensor.
    x = torch.randn(B, C, D, device="cuda", dtype=torch.float32)
    # Our reference normalization is performed on dim=1, but we must reshape x appropriately.
    x_ref = x.clone()
    norm = torch.norm(x_ref, p=2, dim=1, keepdim=True) + 1e-12
    y_ref = x_ref / norm

    kernel_module = build_kernel()
    # The kernel sees the tensor as having total_vectors = numel / C.
    y_kernel = kernel_module.forward(x)
    # Because of incorrect indexing for >2D tensors the result will differ.
    err = (y_kernel - y_ref).abs().max().item()
    assert err > 1e-3, f"Kernel output unexpectedly close to reference in higher-dimensional case, error {err}"

# Issue 4: Test potential shared memory reduction limits by forcing a launch configuration with more threads per block.
# We simulate this by creating a tensor with a large normalization dimension.
# Note: The kernel launch hard-codes threads=256; to trigger the issue one would launch with a blockDim causing >32 warps.
# Here we mimic the scenario by forcing a huge 'C' so that the work distribution requires many warps per block.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_vector_reduction():
    batch_size = 2
    # Choose a very large C such that the inner loop requires many iterations.
    # Note: although blockDim remains 256, more iterations might expose shared memory reduction issues if blockDim was misconfigured.
    dim = 1 << 20  # 1048576 elements per vector
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32)
    kernel_module = build_kernel()
    y_kernel = kernel_module.forward(x)
    y_ref = reference_l2norm(x)
    err = (y_kernel - y_ref).abs().max().item()
    # In a correct kernel the error should be small; if shared memory reduction fails, the error would be large.
    assert err < 1e-3, f"Kernel output error too high with large vector reduction, error {err}"
