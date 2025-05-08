
import torch
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

# Test 1. Triggering the incorrect thread mapping / reduction strategy.
# In a correct implementation, torch.prod should match the CUDA kernel output.
# With the present kernel, the shared memory reduction combines independent full reductions.
def test_incorrect_reduction_mapping():
    my_module = build_kernel()
    # Create an input tensor with a moderately large reduction dimension.
    batch = 4
    dim1 = 128   # reduction dimension length may amplify the error in reduction strategy
    dim2 = 32
    reduction_dim = 1
    x = torch.randn(batch, dim1, dim2, device='cuda', dtype=torch.float32).contiguous()
    # Reference using PyTorch's prod
    reference = torch.prod(x, dim=reduction_dim)
    # Kernel call: note that our custom kernel expects the input and the dimension integer.
    output = my_module.forward(x, reduction_dim)
    torch.cuda.synchronize()
    # The kernel’s output is expected to be incorrect compared to the reference due to mapping error.
    assert not torch.allclose(output, reference, atol=1e-4), \
        "Test did not trigger the incorrect reduction mapping issue as both outputs match."

# Test 2. Triggering the strict float32 type enforcement.
def test_input_tensor_wrong_dtype():
    my_module = build_kernel()
    batch = 4
    dim1 = 64
    dim2 = 64
    reduction_dim = 2
    # Use a tensor of type float64 instead of float32.
    x = torch.randn(batch, dim1, dim2, device='cuda', dtype=torch.float64).contiguous()
    with pytest.raises(RuntimeError) as excinfo:
        _ = my_module.forward(x, reduction_dim)
    assert "must be a CUDA tensor" not in str(excinfo.value), "Unexpected error message"
    # Ideally the error should complain about the tensor type or its incompatibility.

# Test 3. Triggering the contiguous input tensor requirement.
def test_non_contiguous_input():
    my_module = build_kernel()
    batch = 4
    dim1 = 32
    dim2 = 32
    reduction_dim = 0
    x = torch.randn(batch, dim1, dim2, device='cuda', dtype=torch.float32)
    # Make x non-contiguous by transposing
    x_noncontig = x.transpose(0, 1)
    with pytest.raises(RuntimeError) as excinfo:
        _ = my_module.forward(x_noncontig, reduction_dim)
    assert "must be contiguous" in str(excinfo.value), "Kernel did not raise error for non-contiguous input"
