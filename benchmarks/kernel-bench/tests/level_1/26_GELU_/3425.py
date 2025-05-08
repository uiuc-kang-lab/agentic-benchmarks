
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Function to compile and load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="gelu_cuda_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test with tensor of dimension > 2.
def test_tensor_with_three_dimensions():
    my_module = build_kernel()
    # Create a tensor with 3 dimensions.
    x = torch.randn(4, 4, 4, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor|Input tensor must be 1D or 2D"):
        # This call should trigger the TORCH_CHECK on the dimension.
        _ = my_module.forward(x)

# Issue 2: Test with non-contiguous tensor.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a 2D tensor and then transpose it to get a non-contiguous tensor.
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x_t = x.t()  # Transposed tensor (non-contiguous)
    
    # Compute reference output using torch's GELU on the non-contiguous tensor.
    ref = F.gelu(x_t)
    # Call our custom CUDA kernel.
    y = my_module.forward(x_t)
    
    # The kernel assumes a contiguous layout in row-major order.
    # Since x_t is non-contiguous, the result will be wrong.
    # We check that the outputs are NOT close.
    assert not torch.allclose(y, ref, atol=1e-5), (
        "Expected mismatch due to non-contiguous input, but the outputs match."
    )

# Issue 3: Test with a long 1D tensor to stress the fixed 32x32 block configuration.
def test_long_1d_tensor():
    my_module = build_kernel()
    # Create a long 1D tensor.
    N = 100000  # arbitrary size not necessarily a multiple of 32
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    ref = F.gelu(x)
    y = my_module.forward(x)
    
    # If the block configuration were improperly handling edge threads,
    # the output would differ.
    assert torch.allclose(y, ref, atol=1e-5), (
        f"Kernel output differs from torch.nn.functional.gelu output! Max diff: {(y-ref).abs().max()}"
    )

# Issue 4: Test with a CPU tensor to trigger the CUDA tensor check.
def test_cpu_tensor_input():
    my_module = build_kernel()
    x = torch.randn(64, 128, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor"):
        _ = my_module.forward(x)

if __name__ == "__main__":
    pytest.main([__file__])
