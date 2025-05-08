
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

# Issue 1: Incorrect shared-memory reduction when multiple warps per block are active.
# In this test, we choose M so that more than one warp is assigned per block.
def test_incorrect_shared_memory_reduction():
    # Use M such that one block gets multiple warps (blockDim.x/32 = 8 warps per block)
    M = 32  # 32 rows -> at least 4 blocks if warps are mapped to rows or multiple warps in same block
    K = 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, 1, device="cuda", dtype=torch.float32).contiguous()
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Due to interference between warps in shared memory reduction,
    # the kernel output is expected to be incorrect.
    assert not torch.allclose(C, C_ref, atol=1e-3), (
        "The kernel reduction produced a correct result unexpectedly; "
        "expected interference between warps due to shared memory collision."
    )

# Issue 2: Inconsistent use of dynamic shared memory vs. static allocation.
# While not directly causing a runtime error, launching the kernel with dynamic shared memory when
# the kernel uses a fixed-size array may lead to confusion or subtle bugs when adapting kernel parameters.
def test_dynamic_vs_static_shared_memory_usage():
    # We create inputs with sizes that force usage of the dynamic shared memory parameter.
    M = 16
    K = 2048  # large K to ensure the loop iterates many times
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, 1, device="cuda", dtype=torch.float32).contiguous()
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Because the kernel actually uses a fixed-size shared memory (256 elements) regardless
    # of the dynamic shared memory allocation requested, the accumulated sums may be wrong.
    assert not torch.allclose(C, C_ref, atol=1e-3), (
        "The kernel appears to be handling shared memory correctly, "
        "but it was expected to misbehave due to the mismatch between dynamic and static allocations."
    )

# Issue 3: Fixed block size assumption disallows flexibility for different grid configurations.
# This test uses a matrix shape that would, in a more general kernel, require a different block/grid setup.
def test_fixed_block_size_assumption():
    # Choose M such that the number of rows is not an integer multiple of (threads_per_block/32)
    # For a block size of 256 (8 warps), choose an M that forces a partial block.
    M = 10  # less than 8 rows per block would be fine, but the mapping of warps causes irregular index assignments
    K = 512
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, 1, device="cuda", dtype=torch.float32).contiguous()
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # With the hardcoded mapping of warps to rows and static shared memory,
    # the result may be mis-reduced even for partial blocks.
    assert not torch.allclose(C, C_ref, atol=1e-3), (
        "The kernel output for an M that does not match the fixed block mapping is unexpectedly correct."
    )

# Issue 4: Misleading claim about warp-level primitives.
# While this does not necessarily cause wrong numerical results,
# we can at least test that the kernel does not exploit the usually improved performance from warp shuffles.
# We compare performance against a reference implementation using torch.matmul.
# (Note: This is a weak test because performance testing can be noisy on GPUs.)
def test_missing_warp_level_primitive_optimization():
    M = 128
    K = 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, 1, device="cuda", dtype=torch.float32).contiguous()
    my_module = build_kernel()
    # Warm-up calls to rule out JIT overhead
    for _ in range(5):
        _ = my_module.forward(A, B)
    torch.cuda.synchronize()
    import time
    start = time.time()
    for _ in range(20):
        _ = my_module.forward(A, B)
    torch.cuda.synchronize()
    elapsed_kernel = time.time() - start

    start = time.time()
    for _ in range(20):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    elapsed_torch = time.time() - start

    # We expect the custom kernel to at best match but likely be slower than highly optimized torch.matmul,
    # especially since it does not actually use warp-level primitives.
    assert elapsed_kernel > elapsed_torch, (
        "The custom kernel reports performance on par with torch.matmul, "
        "which is unexpected given its lack of proper warp-level primitives."
    )
