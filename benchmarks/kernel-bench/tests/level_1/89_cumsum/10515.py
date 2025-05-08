
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    # Assume kernel.cu is in the same directory as this test file.
    this_dir = os.path.dirname(os.path.realpath(__file__))
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(this_dir, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

# Issue 1: The kernel only supports float32 tensors.
def test_input_tensor_type(kernel_module):
    # Create a tensor of type float64 on CUDA.
    x_double = torch.randn(128, 4000, device="cuda", dtype=torch.float64)
    # The kernel will interpret the data as float32, which is a problem.
    # We check that the output is not equal to the torch.cumsum reference.
    with pytest.raises(Exception):
        # It is likely to crash or produce an error because of type mismatch.
        y = kernel_module.forward(x_double, 1)
        torch.cuda.synchronize()

# Issue 2: The kernel does not handle non-contiguous tensors.
def test_non_contiguous_input(kernel_module):
    # Create a contiguous tensor first.
    x = torch.randn(128, 4000, device="cuda", dtype=torch.float32)
    # Create a non-contiguous view by transposing a 2D tensor (if applicable).
    # Here we first reshape to (128, 4000, 1) and then squeeze at a different dim.
    # Alternatively, we can use .t() on a 2D tensor.
    x_noncontig = x.t()  # This transposition makes the tensor non-contiguous.
    assert not x_noncontig.is_contiguous(), "Test setup error: Tensor is contiguous."

    with pytest.raises(RuntimeError) as excinfo:
        # The kernel should trigger the CHECK_CONTIGUOUS error.
        y = kernel_module.forward(x_noncontig, 0)
    assert "must be contiguous" in str(excinfo.value)
    
# Issue 3: The kernel does not perform any CUDA error checking post launch.
# In some pathological cases the indexing might be off if the memory layout assumptions are violated.
# We'll simulate such a complex case by providing an input with an unexpected stride.
def test_unexpected_memory_layout(kernel_module):
    # Create a tensor with additional batch dimensions and transpose two dimensions.
    # This still returns a contiguous tensor if done properly but can have unexpected stride values.
    x = torch.randn(8, 16, 32, 64, device="cuda", dtype=torch.float32)
    # Transpose two non-adjacent dimensions to create a tensor that is still marked as contiguous? 
    # Actually, most transpositions make a tensor non-contiguous,
    # so we instead do a narrow() operation which preserves contiguity.
    # Here, we simulate a complex layout by first expanding a dimension and then narrowing.
    x_expanded = x.unsqueeze(2).expand(-1, -1, 10, -1, -1)
    # Now take a narrow along the expanded dimension so that our expected 'stride' assumption may not hold.
    x_complex = x_expanded.narrow(2, 0, 5)
    # Although the tensor is contiguous, the physical layout is not the same as a simple outer/inner decomposition.
    # The kernel will use the sizes to compute indices, which will be wrong.
    # We check if the result from our kernel does not match torch.cumsum.
    # (Note: the error might be subtle, so we expect the results to not be equal.)
    # In order to call our kernel, we need to choose a valid dim that is scanned.
    dim_to_scan = 1  # using the second dimension
    y_kernel = kernel_module.forward(x_complex, dim_to_scan)
    # Re-arrange dimensions to match because torch.cumsum takes into account the actual shape.
    y_ref = torch.cumsum(x_complex, dim=dim_to_scan)
    # Instead of allclose, we check that the maximum difference is significant.
    diff = (y_kernel - y_ref).abs().max().item()
    assert diff > 1e-3, f"Kernel unexpectedly matched torch.cumsum despite complex layout (max difference: {diff})"
