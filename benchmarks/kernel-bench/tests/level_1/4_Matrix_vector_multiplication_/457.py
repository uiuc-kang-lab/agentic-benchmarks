
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Atomic Reduction Inefficiency / Non-determinism
# This test creates a scenario where multiple warps are used to reduce a large row.
# Due to the atomicAdd-based reduction, small nondeterministic differences might occur.
# We run the kernel multiple times and check that the outputs, while close to torch.matmul,
# sometimes differ in their least-significant bits.
def test_intra_block_reduction():
    my_module = build_kernel()
    M = 64
    K = 5000  # Large enough to require many iterations per thread
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, 1, device="cuda", dtype=torch.float32)
    
    # Run the kernel twice and compare against torch.matmul
    C1 = my_module.forward(A, B)
    torch.cuda.synchronize()
    C2 = my_module.forward(A, B)
    torch.cuda.synchronize()
    
    C_ref = torch.matmul(A, B)
    
    # Check that both kernel outputs are close to the reference.
    # Allow a slightly looser tolerance due to possible non-determinism.
    assert torch.allclose(C1, C_ref, atol=1e-4), f"Kernel output C1 deviates from reference! Maximum diff: {(C1 - C_ref).abs().max()}"
    assert torch.allclose(C2, C_ref, atol=1e-4), f"Kernel output C2 deviates from reference! Maximum diff: {(C2 - C_ref).abs().max()}"
    
    # Additionally, check that the two kernel runs are not bitwise identical (to indicate nondeterminism).
    # (It is possible but unlikely for nondeterminism to cancel out completely.)
    if torch.equal(C1, C2):
        pytest.skip("Kernel outputs are deterministically equal; unable to trigger atomic reduction nondeterminism test.")

# Issue 2: Lack of Half Precision Support
# This test attempts to run the kernel with half precision inputs.
# The kernel dispatch does not include half (fp16), so we expect an error or incorrect behavior.
def test_half_precision_support():
    my_module = build_kernel()
    M = 32
    K = 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, 1, device="cuda", dtype=torch.float16)
    
    with pytest.raises(RuntimeError):
        # Expect the kernel build/disptach to fail because fp16 is not handled.
        _ = my_module.forward(A, B)
    
# Issue 3: Hard-coded Launch Parameters / Lack of Generality
# This test uses an input where the number of columns K is much smaller than BLOCK_SIZE,
# and the number of rows M is not divisible by the number of streams.
# This checks that the kernel’s assumptions on grid/block dimensions do not break correctness.
def test_launch_configuration_general_input():
    my_module = build_kernel()
    # Choose dimensions that stress the hard-coded BLOCK_SIZE and stream splitting.
    M = 103  # Not divisible by NUM_STREAMS (which is 4 in the kernel)
    K = 128  # Much smaller than BLOCK_SIZE (512)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, 1, device="cuda", dtype=torch.float32)
    
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    
    # We use a strict tolerance because for small K the summation order should have minimal impact.
    assert torch.allclose(C, C_ref, atol=1e-5), f"Kernel output differs from reference output! Max diff: {(C - C_ref).abs().max()}"

if __name__ == '__main__':
    pytest.main([__file__])
