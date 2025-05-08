
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Double-precision atomicAdd problem.
# On GPUs with compute capability < 6.0, atomicAdd for double is not supported.
# This test creates double tensors, runs the kernel, and compares its result with torch.matmul.
# On devices that do not properly support double atomicAdd the kernel output will be off.
def test_double_precision_atomic():
    device = torch.device("cuda")
    cap_major, _ = torch.cuda.get_device_capability(device)
    # Only run if we are on GPU. (Test for double precision atomicAdd misbehavior.)
    M = 32
    K = 1024
    A = torch.randn(M, K, dtype=torch.float64, device=device)
    B = torch.randn(K, 1, dtype=torch.float64, device=device)
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    
    # If the device does not support atomicAdd for double precision (compute capability < 6),
    # then the kernel result is expected to be inaccurate.
    if cap_major < 6:
        assert not torch.allclose(C, C_ref, atol=1e-6), (
            "Test expected a failure in double precision atomicAdd on GPUs with compute capability < 6.0"
        )
    else:
        # For devices that support double precision atomicAdd, we expect correct output.
        assert torch.allclose(C, C_ref, atol=1e-6), (
            f"Kernel output differs from reference output! Max diff: {(C - C_ref).abs().max().item()}"
        )

# Issue 2: PackedTensorAccessor32 index overflow.
# The kernel uses 32-bit indexing, so if a tensor’s dimension exceeds the int32 limit,
# behavior is undefined. Since we cannot really allocate tensors with >2^31 elements,
# we simulate the triggering condition by monkey-patching the shape.
# This test creates a small dummy tensor and then fakes a large K dimension.
@pytest.mark.xfail(reason="Simulated test: cannot allocate huge tensors to trigger 32-bit index overflow")
def test_index_overflow_simulation(monkeypatch):
    device = torch.device("cuda")
    M = 2
    K_real = 128  # Actual allocated K
    # Build a small tensor, then monkey-patch its size to simulate a huge K.
    A = torch.randn(M, K_real, device=device)
    B = torch.randn(K_real, 1, device=device)
    
    # Monkey-patch A.size to simulate an extremely large number of columns.
    original_size = A.size
    def fake_size(dim):
        if dim == 1:
            # Simulate a dimension that exceeds int32 maximum.
            return 2**31  # 2147483648
        return original_size(dim)
    monkeypatch.setattr(A, "size", fake_size)
    
    my_module = build_kernel()
    # Even though the tensor A only has K_real columns allocated,
    # the kernel will use the fake size. This should trigger index access beyond allocated memory.
    with pytest.raises(Exception):
        _ = my_module.forward(A, B)
