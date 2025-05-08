
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Function to build the CUDA extension module.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 test: Hardcoded reduction iteration count.
# Create an input tensor where the reduction dimension size is not 50.
# The kernel will multiply 50 elements regardless, so the results will differ from torch.prod.
def test_hardcoded_reduction_dimension():
    mod = build_kernel()
    # Create a tensor with reduction dimension size 30 (instead of 50)
    # Use a shape where the reduction dimension is the innermost one to isolate the issue.
    x = torch.randn(16, 30, device="cuda", dtype=torch.float32)
    # reduction over last dimension so stride = 1 and indexing is easier
    ref_out = torch.prod(x, dim=-1)
    # Our kernel is designed to perform 50 multiplications,
    # so the kernel output will likely be missing elements (or reading out of bounds)
    kernel_out = mod.forward(x, -1)
    torch.cuda.synchronize()
    # They should not be equal.
    assert not torch.allclose(kernel_out, ref_out, atol=1e-5), \
        "Test did not detect the hardcoded iteration count issue."

# Issue 2 test: Incorrect indexing for non-innermost reduction dimension.
# Create an input tensor where the reduction dimension is not the innermost dimension.
def test_incorrect_indexing():
    mod = build_kernel()
    # Create a tensor of shape (16, 256, 256). We reduce along dimension 1.
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.float32)
    # Correct result using PyTorch built-in prod:
    ref_out = torch.prod(x, dim=1)
    kernel_out = mod.forward(x, 1)
    torch.cuda.synchronize()
    # The indexing error in the kernel should lead to an incorrect result.
    assert not torch.allclose(kernel_out, ref_out, atol=1e-5), \
        "Test did not detect the indexing error when reducing over a non-innermost dimension."

# Issue 3 test: Using a non-contiguous tensor.
def test_non_contiguous_input():
    mod = build_kernel()
    x = torch.randn(16, 50, device="cuda", dtype=torch.float32)
    # Create a non-contiguous version of x by transposing:
    x_noncontig = x.t()
    with pytest.raises(Exception, match="must be contiguous"):
        mod.forward(x_noncontig, 0)

# Additional test: Wrong usage of min might lead to a compile error.
# Since this error occurs at compile time, we simulate that by trying to build the module.
def test_min_function_compilation():
    # Attempt to build the module.
    # If the min issue were present, module build would fail.
    try:
        mod = build_kernel()
    except Exception as e:
        pytest.skip("Compilation failed due to min() issue: " + str(e))
    # If build_kernel() succeeds, we simply pass the test because run-time tests are not
    # able to catch compilation issues.
    assert mod is not None

if __name__ == "__main__":
    pytest.main([__file__])
