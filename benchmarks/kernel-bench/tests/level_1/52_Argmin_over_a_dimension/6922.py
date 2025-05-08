
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Utility function to compile and load the CUDA extension.
def build_kernel():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    module = load(
        name="argmin_cuda",
        sources=[os.path.join(this_dir, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

@pytest.fixture(scope="module")
def cuda_module():
    mod = build_kernel()
    return mod

def test_double_precision_issue(cuda_module):
    # Issue 1: Wrong initialization when using double tensors.
    # Construct a tensor where all values are huge (above FLT_MAX)
    # except one element which is slightly lower. In proper argmin,
    # the minimal value should be found at the correct index.
    # For double type, FLT_MAX (~3.4e38) is far lower than typical double values.
    # Here, we fill with a huge value and then set one element to be slightly lower.
    batch = 4
    K = 10  # reduction dimension length
    other_dim = 3
    # Fill with huge value (e.g., near 1e308) so that no value is less than FLT_MAX.
    huge_val = 1e308
    x = torch.full((batch, K, other_dim), huge_val, dtype=torch.double, device="cuda")
    # Set the minimum in each slice to a value slightly less than huge_val.
    # Place the minimum at index 5 in every slice.
    x[:, 5, :] = huge_val - 1.0

    # The expected indices from torch.argmin over dimension=1
    expected = torch.argmin(x, dim=1)
    # Get the indices computed by our CUDA kernel.
    # Note: The kernel extension expects the reduction dimension as the second argument.
    output = cuda_module.forward(x, 1)
    torch.cuda.synchronize()
    # If the wrong initialization is used, the kernel might never update from FLT_MAX,
    # resulting in an incorrect index (likely 0) for slices.
    assert not torch.equal(output, expected), \
        "Test did not detect the double precision initialization issue."

def test_half_precision_issue(cuda_module):
    # Issue 1 also affects half precision: using FLT_MAX (float max) is not appropriate
    # for half precision (__half). The maximum representable half is around 65504.
    # We design a test where all elements are large (but representable in half)
    # except one element which is slightly lower.
    batch = 4
    K = 10
    other_dim = 3
    # Use values near the upper bound of half (but below 65504) for most entries.
    # To force the issue, we fill with a value that is greater than the half max
    # when interpreted from FLT_MAX initialization.
    high_val = torch.tensor(60000.0).half()  # a high half value but well below 65504 max
    x = torch.full((batch, K, other_dim), high_val, dtype=torch.half, device="cuda")
    # Set the minimum element to a lower value, at index 3.
    x[:, 3, :] = high_val - torch.tensor(100.0, dtype=torch.half)

    expected = torch.argmin(x, dim=1)
    output = cuda_module.forward(x, 1)
    torch.cuda.synchronize()
    # Because the initialization uses FLT_MAX from float,
    # it may corrupt the comparison for half precision values.
    assert not torch.equal(output, expected), \
        "Test did not detect the half precision initialization issue."

def test_block_size_shared_memory_issue(cuda_module):
    # Issue 2: The kernel relies on a hardcoded shared-memory size of 256.
    # We simulate a scenario that forces a mismatch between the number
    # of threads doing work in the reduction and the shared memory allocation.
    #
    # Here we create an input tensor whose reduction dimension (K) is small,
    # so that most threads in the block do not have any valid work. Even if the
    # CUDA launch is fixed to 256 threads as in the current implementation,
    # a general use-case might change the block size to something else.
    #
    # The test will check (by comparing to torch.argmin) if the kernel output
    # is unexpectedly always returning the same index (e.g., 0) due to uncoalesced
    # or out-of-bound shared memory accesses.
    batch = 2
    K = 8  # very small reduction dimension compared to block size of 256
    other_dim = 5
    x = torch.randn(batch, K, other_dim, device="cuda", dtype=torch.float)
    
    expected = torch.argmin(x, dim=1)
    output = cuda_module.forward(x, 1)
    torch.cuda.synchronize()
    # If the shared memory hardcoding is an issue, the reduction may not work correctly.
    assert not torch.equal(output, expected), \
        "Test did not detect the shared memory block size issue."

