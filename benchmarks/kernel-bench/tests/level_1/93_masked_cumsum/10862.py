
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA extension
def build_kernel():
    cuda_module = load(
        name="masked_cumsum_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger shared memory overflow.
# This test constructs an input with a very large size L so that the calculated shared memory size
# exceeds the device limit and the kernel launch should fail.
def test_shared_memory_overflow():
    device = torch.device("cuda")
    prop = torch.cuda.get_device_properties(device)
    # Compute maximum number of bytes available per block for shared memory.
    max_shared = prop.sharedMemPerBlock
    element_size = torch.tensor(0, dtype=torch.float32).element_size()
    bool_size = torch.tensor(True).element_size()
    # Choose L such that L*(element_size + bool_size) is just above the available limit
    L = max_shared // (element_size + bool_size) + 1
    N = 10  # number of rows is small
    x = torch.randn(N, L, device=device, dtype=torch.float32)
    mask = torch.randint(0, 2, (N, L), device=device, dtype=torch.bool)
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise a CUDA error because the dynamic shared memory size exceeds the limit.
        module.forward(x, mask)

# Test 2: Trigger an error due to non–contiguous input tensor.
# The kernel’s host code checks for contiguity. Passing non–contiguous tensors should trigger an error.
def test_non_contiguous_tensor():
    device = torch.device("cuda")
    N, L = 128, 4000
    x = torch.randn(N, L, device=device, dtype=torch.float32)
    mask = torch.randint(0, 2, (N, L), device=device, dtype=torch.bool)
    # Create non–contiguous tensors by transposing
    x_nc = x.transpose(0, 1)
    mask_nc = mask.transpose(0, 1)
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        module.forward(x_nc, mask_nc)

# Test 3: Trigger an error using an invalid dimension for the cumulative sum.
# The kernel host code validates the dimension argument, so this test should catch that.
def test_invalid_dimension():
    device = torch.device("cuda")
    N, L = 128, 4000
    x = torch.randn(N, L, device=device, dtype=torch.float32)
    mask = torch.randint(0, 2, (N, L), device=device, dtype=torch.bool)
    
    module = build_kernel()

    # Use an invalid dim (e.g., 2 for a 2D tensor with dims 0 and 1)
    with pytest.raises(RuntimeError):
        module.forward(x, mask, 2)

# Note: The above tests trigger errors associated with issues 3 and 4 (shared memory size) 
# and host-level checks (non-contiguous inputs and invalid dimensions).
# While issues 1, 2, and 5 (launch configuration mismatch, fixed BLOCK_SIZE, and sequential accumulation)
# do not necessarily crash execution, they are design problems that affect performance and generality.
# In a real-world scenario, additional tests (possibly with performance benchmarks or checking
# parallel correctness) would be needed, but such tests are less straightforward to automate in pytest.
    
if __name__ == "__main__":
    pytest.main([__file__])
