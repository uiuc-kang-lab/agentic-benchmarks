
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to compile and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="prod_reduce_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return cuda_module

# Issue 1 and Issue 2: Incorrect aggregation and wrong pointer arithmetic.
# For a tensor with shape (batch, dim1, dim2) and reducing over dim=1, the expected output
# is torch.prod(x, dim=1). The kernel mis-aggregates across threads, leading to wrong results.
def test_incorrect_aggregation_and_addressing():
    # Create a tensor with clear non-trivial values.
    torch.manual_seed(0)
    batch_size, dim1, dim2 = 4, 8, 5
    reduction_dim = 1
    x = torch.rand(batch_size, dim1, dim2, device="cuda", dtype=torch.float32)
    expected = torch.prod(x, dim=reduction_dim)

    prod_reduce_module = build_kernel()
    # Call the kernel's forward. It expects x and an integer dimension.
    out = prod_reduce_module.forward(x, reduction_dim)
    torch.cuda.synchronize()
    
    # Because the kernel aggregates results incorrectly, the output will be incorrect.
    assert not torch.allclose(out, expected, atol=1e-5), (
        "Test should fail due to incorrect reduction aggregation and pointer arithmetic, "
        "but got matching results."
    )

# Issue 3: Loop unroll makes an assumption on the reduction dimension size.
# Provide an input tensor whose reduction dimension size is not a multiple of 4.
def test_loop_unroll_out_of_bounds():
    # Use a reduction dimension of size 3 (not a multiple of 4).
    batch_size, dim1, dim2 = 2, 3, 4
    reduction_dim = 1
    x = torch.rand(batch_size, dim1, dim2, device="cuda", dtype=torch.float32)
    expected = torch.prod(x, dim=reduction_dim)

    prod_reduce_module = build_kernel()
    out = prod_reduce_module.forward(x, reduction_dim)
    torch.cuda.synchronize()
    
    # The out-of-bounds or mis-unrolled loop may produce a result different from expected.
    assert not torch.allclose(out, expected, atol=1e-5), (
        "Test should fail as the unrolled loop does not handle reduction dim sizes not a multiple of 4."
    )

# Issue 4: Kernel only supports float32; using a double tensor leads to wrong interpretation.
def test_incorrect_dtype():
    batch_size, dim1, dim2 = 4, 5, 6
    reduction_dim = 2
    # Use double precision input deliberately.
    x = torch.rand(batch_size, dim1, dim2, device="cuda", dtype=torch.float64)
    expected = torch.prod(x, dim=reduction_dim)

    prod_reduce_module = build_kernel()
    # The module converts the pointer to float*. This should yield incorrect results.
    out = prod_reduce_module.forward(x, reduction_dim)
    torch.cuda.synchronize()
    
    # The output will differ from the expected due to misinterpretation of data type.
    assert not torch.allclose(out, expected.to(torch.float32), atol=1e-5), (
        "Test should fail because the kernel is not implemented for double precision inputs."
    )

# Issue 5: Lack of error checking for kernel launch.
# We can try to trigger an error by providing a non-contiguous tensor.
def test_non_contiguous_input():
    batch_size, dim1, dim2 = 4, 7, 8
    reduction_dim = 0
    x = torch.rand(batch_size, dim1, dim2, device="cuda", dtype=torch.float32)
    # Make tensor non-contiguous by slicing.
    x_non_contiguous = x.transpose(0, 1)

    prod_reduce_module = build_kernel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        prod_reduce_module.forward(x_non_contiguous, reduction_dim)
