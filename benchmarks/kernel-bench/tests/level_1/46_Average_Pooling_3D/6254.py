
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="pool3d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Ensure that the kernel produces varying outputs.
# We create an input tensor such that different pooling windows yield different averages.
# If the kernel erroneously reuses the first computed sum for all outputs, then all the outputs will be equal.
def test_incorrect_output_values():
    cuda_module = build_kernel()
    
    # Create a tensor with shape (1, 1, 4, 4, 4) with sequentially increasing values.
    # This ensures that distinct pooling windows have different sums.
    x = torch.arange(4*4*4, dtype=torch.float32, device='cuda').reshape(1, 1, 4, 4, 4)
    # Use a kernel size that divides the dimensions in a non-trivial way.
    kernel_size = 2
    stride = 2
    padding = 0
    
    # Compute using the custom CUDA kernel.
    output = cuda_module.forward(x, kernel_size, stride, padding)
    
    # Compute the expected output using PyTorch's AvgPool3d.
    avg_pool = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    expected = avg_pool(x)
    
    # Check dimensions match.
    assert output.shape == expected.shape, f"Output shape {output.shape} does not match expected shape {expected.shape}"
    
    # If the kernel is bugged and reuses the same sum, output will be nearly constant.
    # Therefore, we check that the output is not a constant tensor.
    if torch.allclose(output, output.flatten()[0].expand_as(output)):
        pytest.fail("Kernel output is constant across all elements, which indicates the duplicated loop bug.")
    
    # Also check that the output is close to expected (it should be different if the issue is present).
    if not torch.allclose(output, expected, atol=1e-5):
        pytest.fail("Kernel output does not match expected result, confirming the error from duplicated loops.")

# Test 2: Verify that an input tensor on CPU triggers the device error.
def test_cuda_requirement():
    cuda_module = build_kernel()
    
    # Create a CPU tensor.
    x = torch.randn(1, 1, 4, 4, 4, dtype=torch.float32, device='cpu')
    kernel_size = 2
    stride = 2
    padding = 0
    
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor"):
        # This should raise an error due to the TORCH_CHECK in the kernel.
        cuda_module.forward(x, kernel_size, stride, padding)
