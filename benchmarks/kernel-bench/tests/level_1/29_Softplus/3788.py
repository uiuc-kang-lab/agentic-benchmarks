
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel():
    cuda_module = load(
        name="softplus_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue with large input (grid dimension limit)
def test_large_input():
    # Create an input tensor larger than 65535*512 elements
    # For example, create a tensor with (65535 * 512 + 1000) elements
    num_elements = 65535 * 512 + 1000
    # Use a 1D tensor for simplicity
    x = torch.randn(num_elements, device="cuda", dtype=torch.float32)
    module = build_kernel()
    out_cuda = module.forward(x)
    # Compare with PyTorch's softplus
    out_ref = F.softplus(x)
    # The kernel may produce wrong results on the extra elements if grid limit is hit.
    assert torch.allclose(out_cuda, out_ref, atol=1e-4), (
        f"Kernel output incorrect for large input size; max difference: "
        f"{(out_cuda - out_ref).abs().max().item()}"
    )

# Test 2: Trigger issue with non-contiguous input (lack of contiguity check)
def test_non_contiguous_input():
    # Create a 2D tensor and take a slice to make it non-contiguous.
    x = torch.randn(128, 128, device="cuda", dtype=torch.float32)
    x_nc = x[:, ::2]  # non-contiguous slice
    module = build_kernel()
    # Even though data_ptr may still return valid pointer,
    # noncontiguous inputs may lead to incorrect results.
    out_cuda = module.forward(x_nc)
    out_ref = F.softplus(x_nc)
    # This test verifies if the kernel works on a non-contiguous tensor.
    assert torch.allclose(out_cuda, out_ref, atol=1e-4), (
        f"Kernel output incorrect for non-contiguous input; max difference: "
        f"{(out_cuda - out_ref).abs().max().item()}"
    )

# Test 3: Trigger issue by passing a CPU tensor (lack of device check)
def test_cpu_input_error():
    x = torch.randn(1024, dtype=torch.float32, device="cpu")
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect runtime error because kernel expects CUDA tensor.
        module.forward(x)

# Test 4: Trigger issue with unsupported half precision type
def test_half_precision_error():
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Depending on PyTorch version/dispatch macros, the half type might not be supported.
        module.forward(x)
        
if __name__ == "__main__":
    pytest.main([__file__])
