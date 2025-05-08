
import pytest
import torch
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

# Issue 1: Lack of error checking after kernel launch.
# We trigger this by providing a tensor on the wrong device (CPU) to the CUDA kernel.
# This should cause a runtime error because the kernel launch will receive a pointer
# not allocated on CUDA device.
def test_invalid_device():
    my_module = build_kernel()
    # Create a CPU tensor even though our kernel expects a CUDA tensor.
    cpu_tensor = torch.randn(1024, dtype=torch.float32)  # CPU tensor
    with pytest.raises(RuntimeError):
        # This should produce a runtime error since the underlying kernel expects a CUDA pointer.
        my_module.forward(cpu_tensor)

# Issue 2: Non-contiguous memory.
# The kernel assumes contiguous memory and uses a linear indexing over numel().
# Passing a non-contiguous tensor can result in wrong outputs.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x = torch.randn(64, 32, dtype=torch.float32, device="cuda")
    non_contiguous = x.t()  # transpose: non-contiguous view
    # Run the CUDA kernel on the non-contiguous input
    output = my_module.forward(non_contiguous)
    # Compute the expected result using torch.relu on the non-contiguous tensor
    # Note: torch.relu operates element-wise and respects the tensor’s layout.
    expected = torch.relu(non_contiguous)
    # The output from the kernel is computed using a linear traversal over memory.
    # This test should fail if the kernel incorrectly assumes contiguous memory.
    assert not torch.allclose(output, expected), "Kernel should fail for non-contiguous inputs."

# Issue 3: Lack of support for non-floating point types.
# The kernel dispatch macro only accepts floating types. If a user passes an integer tensor,
# the AT_DISPATCH_FLOATING_TYPES macro will not match and should raise an error.
def test_unsupported_dtype():
    my_module = build_kernel()
    # Create an integer tensor on CUDA.
    int_tensor = torch.randint(low=-100, high=100, size=(1024,), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        my_module.forward(int_tensor)
        
if __name__ == "__main__":
    pytest.main([__file__])
