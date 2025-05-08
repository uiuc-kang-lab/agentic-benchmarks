
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="leaky_relu_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case for Issue 1:
# Passing a non-float32 tensor (float64) to the CUDA kernel.
def test_non_float32_input():
    # Create a float64 tensor on CUDA.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    negative_slope = 0.01
    module = build_kernel()

    # The kernel does not dispatch based on dtype: it assumes float. As a result, the memory will be
    # misinterpreted. We compute a reference using PyTorch's leaky_relu on double.
    ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    
    # Run the CUDA kernel. Even though x is double, the kernel will treat its bytes as float.
    y = module.forward(x)
    
    # Since the kernel misinterprets the double data (8 bytes per element vs the kernel expecting 4 bytes),
    # the output will not match the reference.
    # We check that the difference is unacceptable.
    assert not torch.allclose(y.cpu().to(torch.float64), ref, atol=1e-3), \
        "Kernel incorrectly accepted non-float32 input without producing an error."

# Test case for Issue 2:
# Passing a non-contiguous tensor causing the CHECK_CONTIGUOUS macro to trigger.
def test_non_contiguous_input():
    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x = torch.randn(128, 64, device="cuda", dtype=torch.float32)
    x_noncontiguous = x.t()  # transpose makes it non-contiguous
    negative_slope = 0.01
    module = build_kernel()
    
    with pytest.raises(RuntimeError, match="must be contiguous"):
        # This call should raise an error because the tensor is not contiguous.
        _ = module.forward(x_noncontiguous)
        
if __name__ == "__main__":
    pytest.main([__file__])
