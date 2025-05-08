
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="tanh_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Non-contiguous tensor causes potential misaligned vectorized loads.
def test_non_contiguous_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Create a contiguous tensor and then make it non-contiguous via transpose.
    # The vectorized kernel assumes proper alignment, so a non-contiguous tensor
    # may lead to incorrect results.
    N, M = 16, 17  # arbitrary sizes
    x = torch.randn(N, M, dtype=torch.float32, device="cuda")
    # Make the tensor non-contiguous
    x = x.t()  
    # Compute reference using PyTorch tanh
    x_ref = torch.tanh(x)
    
    my_module = build_kernel()
    y = my_module.forward(x)
    torch.cuda.synchronize()
    
    # We expect that the kernel, which does not check contiguity, might produce an incorrect result.
    assert not torch.allclose(y, x_ref, atol=1e-5), \
        "Kernel did not expose misalignment issue on non-contiguous input."

# Issue 2: Kernel does not support half precision tensors.
def test_half_precision_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Create a half precision tensor. The AT_DISPATCH_FLOATING_TYPES macro does not include float16.
    x = torch.randn(1024, dtype=torch.float16, device="cuda")
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # We expect the dispatch to fail for half precision because it is not handled.
        my_module.forward(x)

# Issue 3: Lack of error checking after kernel launch might hide runtime errors.
# One way to force a runtime error is to pass a CPU tensor to a CUDA kernel.
def test_cpu_tensor_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Create a CPU tensor
    x = torch.randn(1024, dtype=torch.float32, device="cpu")
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Passing a CPU tensor should trigger an error in the CUDA kernel execution.
        my_module.forward(x)
        
if __name__ == "__main__":
    pytest.main([__file__])
