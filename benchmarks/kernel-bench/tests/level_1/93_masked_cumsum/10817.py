
import torch
import pytest
from torch.utils.cpp_extension import load
import threading

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper function: reference masked cumulative sum
def ref_masked_cumsum(x, mask, dim):
    return torch.cumsum(x * mask, dim=dim)

# Issue 1: Testing concurrent kernel launches that may conflict due to constant memory usage.
def test_concurrent_kernel_launches():
    cuda_module = build_kernel()

    # Create two inputs with different last dimensions to force different d_L values.
    batch_size = 8

    # Input A with L = 30 (parallel branch will be used)
    L_A = 30
    x_A = torch.randn(batch_size, L_A, device='cuda')
    mask_A = torch.randint(0, 2, (batch_size, L_A), device='cuda', dtype=torch.bool)

    # Input B with L = 100 (will likely trigger the fallback sequential branch)
    L_B = 100
    x_B = torch.randn(batch_size, L_B, device='cuda')
    mask_B = torch.randint(0, 2, (batch_size, L_B), device='cuda', dtype=torch.bool)

    # Launch the two kernel invocations concurrently in separate streams.
    stream_A = torch.cuda.Stream()
    stream_B = torch.cuda.Stream()

    output_A = [None]
    output_B = [None]

    def launch_A():
        with torch.cuda.stream(stream_A):
            output_A[0] = cuda_module.forward(x_A, mask_A)
            stream_A.synchronize()

    def launch_B():
        with torch.cuda.stream(stream_B):
            output_B[0] = cuda_module.forward(x_B, mask_B)
            stream_B.synchronize()

    thread_A = threading.Thread(target=launch_A)
    thread_B = threading.Thread(target=launch_B)
    thread_A.start()
    thread_B.start()
    thread_A.join()
    thread_B.join()

    # Compute correct cumulative sums on CPU (or using PyTorch) to compare.
    ref_A = ref_masked_cumsum(x_A, mask_A, dim=-1)
    ref_B = ref_masked_cumsum(x_B, mask_B, dim=-1)

    assert torch.allclose(output_A[0], ref_A, atol=1e-5), "Concurrent launch: Output A mismatches reference."
    assert torch.allclose(output_B[0], ref_B, atol=1e-5), "Concurrent launch: Output B mismatches reference."

# Issue 2: Testing small L values which may result in a block size that is not a full warp.
def test_small_length_scan():
    cuda_module = build_kernel()
    
    # Choose a very small L that forces block dim to be next_power_of_2(L) which is less than 32.
    batch_size = 8
    L = 3  # very small, next_power_of_2(3)==4, not a full warp (32)
    x = torch.randn(batch_size, L, device='cuda')
    mask = torch.randint(0, 2, (batch_size, L), device='cuda', dtype=torch.bool)

    output = cuda_module.forward(x, mask)
    ref = ref_masked_cumsum(x, mask, dim=-1)
    # This test may fail if the warp-level scan accesses incorrect data due to using an improper shuffle mask.
    assert torch.allclose(output, ref, atol=1e-5), "Small length scan: Output does not match reference."

# Issue 3: Testing that block configuration may be faulty when L is not a multiple of warp size.
def test_non_multiple_of_warp_size():
    cuda_module = build_kernel()

    # Choose L that is not a multiple of the warp size but small enough to use the parallel branch.
    # E.g., if L=45 then next_power_of_2(45)==64 (which is 2 full warps) but only 45 threads will be valid.
    batch_size = 8
    L = 45
    x = torch.randn(batch_size, L, device='cuda')
    mask = torch.randint(0, 2, (batch_size, L), device='cuda', dtype=torch.bool)

    output = cuda_module.forward(x, mask)
    ref = ref_masked_cumsum(x, mask, dim=-1)
    # The incorrect active mask in the warp-level scan can lead to wrong cumulative values.
    assert torch.allclose(output, ref, atol=1e-5), "Non-multiple of warp size: Output does not match reference."

if __name__ == "__main__":
    pytest.main([__file__])
