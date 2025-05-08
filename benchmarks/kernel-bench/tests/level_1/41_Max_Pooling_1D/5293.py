
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="maxpool1d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Kernel only supports float32.
def test_non_float32_input():
    cuda_module = build_kernel()
    # Create a double tensor (float64) which should trigger a failure due to the type check
    x = torch.randn(2, 4, 16, dtype=torch.double, device="cuda")
    with pytest.raises(RuntimeError):
        # This should trigger TORCH_CHECK or a device error because x.is_cuda() passes but the kernel expects float*
        cuda_module.forward(x, 3, 1, 0, 1, False)

# Issue 2: Incorrect behavior when return_indices is True.
def test_return_indices_concatenation():
    cuda_module = build_kernel()
    # Create a float tensor for which the run should produce an output with concatenated indices.
    # Standard PyTorch MaxPool1d returns a tuple (output, indices). Here our kernel returns a single tensor.
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = True
    x = torch.randn(2, 4, 16, dtype=torch.float32, device="cuda")
    
    output = cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    # The expected behavior for return_indices True is to return two tensors.
    # The concatenated output here will have last dimension of size 2 * output_length.
    expected_output_length = ((x.size(2) + 2 * padding - dilation*(kernel_size-1) - 1) // stride) + 1
    # So if the implementation were correct (returning a tuple), the output shape would be (2, 4, expected_output_length).
    # Instead, we check for the incorrect concatenation behavior.
    assert output.size(-1) == 2 * expected_output_length, \
        "Kernel returned concatenated output instead of separate (output, indices) tensors when return_indices is True."

# Issue 3: The kernel requires contiguous float32 input. Test non-contiguous input.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    x = torch.randn(2, 4, 16, dtype=torch.float32, device="cuda")
    # Make x non contiguous by transposing last two dimensions (although for a 3D tensor, transposing channels and length)
    x_noncontig = x.transpose(1, 2)
    # The kernel's TORCH_CHECK in forward should trigger here.
    with pytest.raises(RuntimeError):
        cuda_module.forward(x_noncontig, 3, 1, 0, 1, False)

# Issue 4: No kernel launch error checking. Trigger by providing parameters that yield an invalid output length.
def test_invalid_output_length():
    cuda_module = build_kernel()
    # By choosing parameters that result in negative output length, we trigger a TORCH_CHECK failure.
    x = torch.randn(2, 4, 10, dtype=torch.float32, device="cuda")
    # Set parameters to cause negative effective output length (e.g., kernel_size too large with dilation)
    kernel_size = 5
    stride = 1
    padding = 0
    dilation = 3  # This will make (input_length + 2*padding - dilation*(kernel_size-1) -1) negative.
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, kernel_size, stride, padding, dilation, False)
