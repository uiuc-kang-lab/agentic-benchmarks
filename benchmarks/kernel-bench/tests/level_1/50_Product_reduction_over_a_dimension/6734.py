
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension (assumes kernel.cu is in the same directory)
def build_kernel():
    # Ensure that CUDA is available.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel works only for float32. Test using a non-float32 (double) tensor.
def test_input_tensor_type():
    my_module = build_kernel()
    # Create a double tensor (float64); note: our kernel assumes float32.
    x = torch.randn(16, 256, 256, dtype=torch.float64, device='cuda')
    dim = 1
    # The kernel code does not check for dtype.
    # We expect that the result will differ from torch.prod (or even be garbage).
    output_kernel = my_module.forward(x, dim)
    output_ref = torch.prod(x, dim=dim)
    # Since the kernel incorrectly casts the double pointer to float*,
    # the result will be much different than reference.
    with pytest.raises(AssertionError):
        assert torch.allclose(output_kernel, output_ref, atol=1e-5), \
            "Kernel unexpectedly produced correct output for a non float32 input"

# Issue 2: Kernel requires contiguous input.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    x = torch.randn(16, 256, 256, device='cuda', dtype=torch.float32)
    # Make x non-contiguous by transposing
    x_noncontig = x.transpose(1, 2)
    dim = 2  # choose a valid dim
    with pytest.raises(RuntimeError) as excinfo:
        _ = my_module.forward(x_noncontig, dim)
    # Expect an error message containing "must be contiguous"
    assert "must be contiguous" in str(excinfo.value)

# Issue 3: Load imbalance / remainder handling.
def test_load_imbalance():
    my_module = build_kernel()
    # Choose dimensions such that output.numel() is not an exact multiple of elements_per_thread.
    # For example, use a reduction dimension whose size is not a multiple of 4 and an output with a small numel.
    # Here, input shape (2, 7, 5) and reducing dim=1 results in an output of shape (2, 5) i.e. 10 elements.
    # The kernel is configured with elements_per_thread=4 so 10 is not divisible by 4.
    x = torch.randn(2, 7, 5, device='cuda', dtype=torch.float32)
    dim = 1
    output_kernel = my_module.forward(x, dim)
    output_ref = torch.prod(x, dim=dim)
    # Even though the remainder work is assigned to a single thread,
    # the result (if correct) should match torch.prod.
    assert torch.allclose(output_kernel, output_ref, atol=1e-5), \
        "Kernel output does not match torch.prod in a load imbalance/remainder situation"

# Issue 4: Use of int for indexing may cause overflow for large tensors.
@pytest.mark.skip(reason="This test simulates an overflow scenario which is impractical to run on current hardware")
def test_integer_overflow():
    my_module = build_kernel()
    # Construct dimensions such that the total number of elements would exceed the maximum value of int.
    # Note: This test is illustrative. In practice, one cannot allocate a tensor with > 2^31 elements.
    # For example, simulate a tensor with shape that would yield a huge linear index range.
    # Here, we simply create a tensor with a moderately large size and check that the result is incorrect if overflow occurs.
    # In a proper implementation, one would use int64 for indexing.
    huge = 2**30  # a huge dimension size (this is not allocatable; used for simulation)
    # Instead, we simulate by monkey patching the kernel parameters. For testing purposes, we set num_elements close to int max.
    # For example, create a tensor with shape (1, 100000, 1) so that output.numel() is 1*1 = 1.
    # Then, manually set dim_size to a huge value to simulate internal overflow.
    x = torch.randn(1, 100000, 1, device='cuda', dtype=torch.float32)
    dim = 1
    # WARNING: This simulation does not trigger a real overflow but acts as a placeholder for the scenario.
    output_kernel = my_module.forward(x, dim)
    output_ref = torch.prod(x, dim=dim)
    # In an overflow scenario, the output would be incorrect.
    # Here we assert that the kernel output is likely incorrect.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), \
        "Kernel output unexpectedly matches torch.prod even though an integer overflow scenario was simulated"

if __name__ == "__main__":
    pytest.main([__file__])
