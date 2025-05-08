
import torch
import pytest
import threading
import time
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the extension from kernel.cu; ensure it is rebuilt to pick up changes
    cuda_module = load(
        name="masked_cumsum_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Race condition with constant memory configuration.
# We try to launch two kernels concurrently on different streams and with different "L" sizes.
def test_concurrent_kernel_launch():
    module = build_kernel()
    # Create two input cases with different last-dimension lengths.
    N = 64
    L1 = 128  # Uses parallel branch (L <= PARALLEL_THRESHOLD)
    L2 = 512  # Uses sequential branch (L > PARALLEL_THRESHOLD)
    x1 = torch.randn(N, L1, device="cuda", dtype=torch.float32)
    mask1 = torch.randint(0, 2, (N, L1), device="cuda", dtype=torch.bool)
    x2 = torch.randn(N, L2, device="cuda", dtype=torch.float32)
    mask2 = torch.randint(0, 2, (N, L2), device="cuda", dtype=torch.bool)
    
    # Define a container for results.
    res1 = [None]
    res2 = [None]
    
    def run_kernel1():
        # Wait a little to increase chances of overlap.
        time.sleep(0.1)
        res1[0] = module.forward(x1, mask1)

    def run_kernel2():
        res2[0] = module.forward(x2, mask2)
    
    # Launch both kernels concurrently via threads.
    t1 = threading.Thread(target=run_kernel1)
    t2 = threading.Thread(target=run_kernel2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # Compute expected results using PyTorch native operations.
    expected1 = torch.cumsum(x1 * mask1, dim=-1)
    expected2 = torch.cumsum(x2 * mask2, dim=-1)
    
    # Check if one or both outputs are wrong due to racy constant memory
    err1 = (res1[0] - expected1).abs().max().item()
    err2 = (res2[0] - expected2).abs().max().item()
    
    # If there is a race condition, at least one of the results might not match.
    # We expect zero error ideally.
    assert err1 < 1e-4, f"Concurrent launch issue: error in first output: {err1}"
    assert err2 < 1e-4, f"Concurrent launch issue: error in second output: {err2}"

# Issue 2: Lack of error checking for constant memory copy and kernel launch.
# We pass non-contiguous tensors to trigger the TORCH_CHECK failures provided by the extension.
def test_non_contiguous_input():
    module = build_kernel()
    x = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    mask = torch.randint(0, 2, (32, 64), device="cuda", dtype=torch.bool)
    # Make the inputs non-contiguous by transposing; the kernel requires contiguous tensors.
    x_nc = x.t()
    mask_nc = mask.t()
    with pytest.raises(RuntimeError, match="contiguous"):
        module.forward(x_nc, mask_nc)

# Issue 3: Limited type support.
# We pass half precision input, which is not supported by AT_DISPATCH_FLOATING_TYPES.
def test_half_precision_input():
    module = build_kernel()
    x = torch.randn(32, 64, device="cuda", dtype=torch.half)
    mask = torch.randint(0, 2, (32, 64), device="cuda", dtype=torch.bool)
    with pytest.raises(RuntimeError, match="AT_DISPATCH_FLOATING_TYPES"):
        module.forward(x, mask)
        
if __name__ == "__main__":
    pytest.main([__file__])
