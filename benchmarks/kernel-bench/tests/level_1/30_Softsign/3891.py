
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu.
    cuda_module = load(
        name="softsign_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_input_dtype_issue():
    # Issue 1: If the input tensor is not float32, the kernel will misinterpret the data.
    # Here we create a double tensor and compute the reference softsign using PyTorch.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # Compute reference softsign with correct dtype (float64)
    softsign_ref = x / (1.0 + torch.abs(x))
    
    # Load the kernel. The kernel, however, expects a float* (i.e. float32)
    kernel = build_kernel()
    # Call the kernel with a double tensor. Since there is no check for type,
    # the kernel will simply reinterpret the memory incorrectly.
    out = kernel.forward(x)
    torch.cuda.synchronize()
    
    # Cast kernel output to float64 for the comparison.
    out_cast = out.to(dtype=torch.float64)
    # The output should be wrong because the kernel interpreted the 64-bit data as 32-bit.
    # We assert that the kernel output is significantly different from the reference.
    assert not torch.allclose(out_cast, softsign_ref, atol=1e-5), \
        "Kernel unexpectedly handled non-float32 input correctly, but it should not."

def test_cpu_tensor_issue():
    # Issue 2: The kernel’s CHECK_CUDA macro ensures the input is a CUDA tensor.
    # If a CPU tensor is passed, a RuntimeError should be raised.
    x_cpu = torch.randn(1024, device="cpu", dtype=torch.float32)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        kernel.forward(x_cpu)
        
if __name__ == "__main__":
    pytest.main([__file__])
