
import torch
import math
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

# A reference GELU implementation (matching the kernel's intent)
def gelu_ref(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))

# Test 1: Trigger the vectorized load misalignment issue by using an input tensor whose total number
#           of elements is not divisible by 4 so that the kernel takes the fallback branch.
def test_vectorized_load_boundary():
    # Create a tensor with a number of elements that is not divisible by 4.
    # For instance, use a 1D tensor of length 1023.
    n = 1023
    x = torch.randn(n, device="cuda", dtype=torch.float32).contiguous()
    
    module = build_kernel()
    y = module.forward(x)
    torch.cuda.synchronize()
    
    y_ref = gelu_ref(x)
    
    # We expect the output to be nearly equal, but due to mis-loading or uninitialized memory,
    # there might be large discrepancies.
    assert not torch.allclose(y, y_ref, atol=1e-4), (
        "Test should trigger boundary misalignment issues: output unexpectedly matches reference."
    )

# Test 2: Trigger the uninitialized shared memory issue.
#           Use an input tensor size such that not every thread performs a valid load.
def test_uninitialized_shared_memory():
    # Choose a size that is not a multiple of BLOCK_SIZE*ITEMS_PER_THREAD.
    # BLOCK_SIZE is 256 and ITEMS_PER_THREAD is 4, so try a size that is one less.
    block_total = 256 * 4
    n = block_total - 1  # 1023 elements
    x = torch.randn(n, device="cuda", dtype=torch.float32).contiguous()
    
    module = build_kernel()
    y = module.forward(x)
    torch.cuda.synchronize()
    
    y_ref = gelu_ref(x)
    
    # Expect differences because some shared memory locations might remain uninitialized.
    assert not torch.allclose(y, y_ref, atol=1e-4), (
        "Test should reveal uninitialized memory reads: output unexpectedly matches reference."
    )

# Test 3: Trigger alignment issues.
#           Create a non-contiguous tensor or a tensor with an offset, so that its data pointer
#           may not be aligned to 16 bytes for safe reinterpret_cast to float4.
def test_alignment_issue():
    # Create a large tensor and then slice it to produce a misaligned view.
    x_full = torch.randn(1024 + 1, device="cuda", dtype=torch.float32)
    # Slicing this tensor might result in a non-aligned data pointer.
    x = x_full[1:].contiguous()  # Likely misaligned with respect to 16 bytes.
    
    module = build_kernel()
    # Our kernel requires the tensor be contiguous and on CUDA, but does not check alignment.
    # This may trigger misaligned accesses in the kernel.
    y = module.forward(x)
    torch.cuda.synchronize()
    
    y_ref = gelu_ref(x)
    
    # Expect differences due to potential misalignment issues.
    assert not torch.allclose(y, y_ref, atol=1e-4), (
        "Test should reveal alignment issues: output unexpectedly matches reference."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
