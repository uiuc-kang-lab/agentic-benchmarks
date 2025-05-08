
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np

def build_kernel():
    # Force recompilation to pick up any changes.
    cuda_module = load(
        name="rms_norm_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function: RMS normalization as in the PyTorch code.
def rms_norm_torch(x, eps):
    # Assuming normalization along dim 1
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    return x / rms

# Issue 1: Non-contiguous tensor
def test_non_contiguous_input():
    kernel_module = build_kernel()
    batch_size = 4
    num_features = 8
    # Create a contiguous tensor and then a non-contiguous view via transpose
    x = torch.randn(batch_size, num_features, 32, 32, device="cuda")
    x = x.transpose(1, 2)  # Now shape is (batch_size, 32, num_features, 32) and non-contiguous.
    
    # Our kernel assumes contiguous memory with shape (B, F, ...)
    # If non-contiguous, the indexing will be wrong.
    try:
        out_cuda = kernel_module.forward(x, 1e-5)
    except Exception as e:
        # If an error is thrown we consider this test passed because the kernel
        # incorrectly assumes contiguous input.
        pytest.skip("Kernel does not handle non-contiguous tensors, as expected.")

    out_torch = rms_norm_torch(x, 1e-5)
    
    # The outputs will not be close if the indexing is off.
    assert not torch.allclose(out_cuda, out_torch, atol=1e-4), "Kernel incorrectly handled non-contiguous input."

# Issue 2: __constant__ memory race condition with different eps values.
def test_eps_constant_memory_race():
    kernel_module = build_kernel()
    batch_size = 4
    num_features = 16
    # Create a standard contiguous tensor.
    x = torch.randn(batch_size, num_features, 16, 16, device="cuda")
    
    # Launch two kernels in succession with different eps values.
    # Because d_eps is in constant memory and is global,
    # if concurrency is not handled correctly the second call might override the first.
    eps1 = 1e-5
    eps2 = 1e-3
    
    out1 = kernel_module.forward(x, eps1)
    out2 = kernel_module.forward(x, eps2)
    
    # Compute expected outputs with torch.
    expected1 = rms_norm_torch(x, eps1)
    expected2 = rms_norm_torch(x, eps2)
    
    # Due to the design flaw, one of the outputs could be computed with the wrong eps.
    mismatch1 = not torch.allclose(out1, expected1, atol=1e-4)
    mismatch2 = not torch.allclose(out2, expected2, atol=1e-4)
    
    assert mismatch1 or mismatch2, ("Kernel did not exhibit the expected issue with "
                                     "__constant__ eps in concurrent calls.")

# Issue 3: Lack of error checking – passing a CPU tensor should cause a runtime error.
def test_cpu_tensor_error():
    kernel_module = build_kernel()
    batch_size = 4
    num_features = 8
    x = torch.randn(batch_size, num_features, 16, 16, device="cpu")
    with pytest.raises(RuntimeError):
        # This should raise because the kernel expects a CUDA tensor.
        kernel_module.forward(x, 1e-5)

# Issue 4: fp16 support may be incorrect.
def test_half_precision():
    kernel_module = build_kernel()
    batch_size = 4
    num_features = 16
    x = torch.randn(batch_size, num_features, 16, 16, device="cuda").half()
    
    out_cuda = kernel_module.forward(x, 1e-5)
    expected = rms_norm_torch(x, 1e-5)
    
    # Allow for larger tolerance because half precision has lower accuracy.
    # If the result is wildly off it indicates sqrt for half is not working as expected.
    assert not torch.allclose(out_cuda, expected, atol=1e-2), (
        "Kernel appears to handle half precision too well. Expected mis-handling due to lack of special support."
    )
