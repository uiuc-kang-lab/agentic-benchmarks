
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="softplus_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Triggering issue with incorrect shared memory size
# Use a double tensor (float64) which will require shared memory size = threads * sizeof(double)
def test_shared_memory_allocation_double():
    # Create input tensor of type double
    batch_size, dim = 16, 16384
    x = torch.randn(batch_size, dim, dtype=torch.float64, device="cuda")
    module = build_kernel()
    # Run the kernel; this may crash or produce incorrect output because the shared memory allocation is wrong.
    with pytest.raises(Exception):
        # We expect an error or misbehavior, so an exception is acceptable.
        out = module.forward(x)
        # Force synchronization to trigger potential kernel errors
        torch.cuda.synchronize()

# Test case 2: Triggering issue with non-contiguous input tensors
def test_non_contiguous_input():
    # Create input tensor and make it non-contiguous through transposition.
    batch_size, dim = 16, 16384
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32)
    x_noncontiguous = x.t()  # Transpose makes it non-contiguous
    
    module = build_kernel()
    # The kernel expects a contiguous pointer. This may lead to incorrect computations.
    out = module.forward(x_noncontiguous)
    torch.cuda.synchronize()
    
    # For reference, compute the expected output using PyTorch's softplus
    # Here, we must also transpose back the output if needed.
    out_ref = torch.nn.functional.softplus(x_noncontiguous)
    
    # The outputs may not be equal because the kernel did not account for non-contiguous memory,
    # therefore we check that they are significantly different.
    # If they are equal, then the issue is not triggered.
    if torch.allclose(out, out_ref, atol=1e-5):
        pytest.fail("Kernel processed non-contiguous input as if it were contiguous.")

