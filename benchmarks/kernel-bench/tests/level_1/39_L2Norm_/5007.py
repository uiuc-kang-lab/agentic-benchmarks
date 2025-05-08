
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="l2norm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def reference_l2norm(x: torch.Tensor) -> torch.Tensor:
    # Torch reference normalization along dim=1, mimicking the original model.forward.
    # This assumes a normalization on dimension 1.
    norm = x.norm(p=2, dim=1, keepdim=True)
    return x / (norm + 1e-12)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shared_memory_overrun():
    # Issue 1: Trigger configuration that forces blocks_per_vector > 32.
    # We need a very large C so that (C + SEGMENT_SIZE - 1) / SEGMENT_SIZE > 32.
    # Given SEGMENT_SIZE = 1024, choose C > 32*1024 = 32768.
    C = 33000
    batch_size = 4  # low batch size, but each vector is huge.
    # Create a contiguous float tensor.
    x = torch.randn(batch_size, C, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    # Call the forward CUDA kernel.
    out = my_module.forward(x)
    torch.cuda.synchronize()
    out_ref = reference_l2norm(x)
    # Due to shared memory reduction overflow, we expect a significant difference.
    diff = (out - out_ref).abs().max().item()
    assert diff < 1e-4, f"Shared memory reduction error triggered; max diff: {diff}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_atomicAdd_double_type():
    # Issue 2: Test with double precision input.
    batch_size, C = 16, 2048
    x = torch.randn(batch_size, C, device="cuda", dtype=torch.float64)
    my_module = build_kernel()
    out = my_module.forward(x)
    torch.cuda.synchronize()
    out_ref = reference_l2norm(x)
    # With double type, atomicAdd may not work correctly on devices without proper support.
    diff = (out - out_ref).abs().max().item()
    assert diff < 1e-6, f"Atomic add error for double type; max diff: {diff}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multi_dimensional_input():
    # Issue 3: Test with a 3D tensor.
    # The kernel assumes normalization on dim=1 but a 3D tensor has a different layout.
    batch_size, C, extra = 8, 1024, 5
    x = torch.randn(batch_size, C, extra, device="cuda", dtype=torch.float32)
    # Even though the forward in the CUDA kernel is written for dim=1,
    # when used with a 3D tensor, the computed outer_stride and total_vectors will be wrong.
    my_module = build_kernel()
    out = my_module.forward(x)
    torch.cuda.synchronize()
    # Compute reference based on normalization along dim=1 for each 2D slice
    norm = x.norm(p=2, dim=1, keepdim=True)
    out_ref = x / (norm + 1e-12)
    diff = (out - out_ref).abs().max().item()
    assert diff < 1e-4, f"Kernel failed on multi-dimensional input; max diff: {diff}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    # Issue 4: Test with a noncontiguous tensor.
    batch_size, C = 16, 2048
    # Create a tensor and then transpose to create noncontiguous layout.
    x = torch.randn(batch_size, C, device="cuda", dtype=torch.float32)
    x = x.t()  # This makes the tensor noncontiguous and changes the stride layout.
    my_module = build_kernel()
    out = my_module.forward(x)
    torch.cuda.synchronize()
    # The reference normalization may need to be computed on the original layout.
    # Here, we mimic the expected behavior (still normalizing along dim=1 of the noncontiguous tensor).
    norm = x.norm(p=2, dim=1, keepdim=True)
    out_ref = x / (norm + 1e-12)
    diff = (out - out_ref).abs().max().item()
    assert diff < 1e-4, f"Kernel failed on noncontiguous input; max diff: {diff}"
