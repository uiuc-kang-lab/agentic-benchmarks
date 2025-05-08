
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the CUDA extension from the file kernel.cu.
    return load(
        name="hardsigmoid_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

@pytest.fixture(scope="module")
def cuda_module():
    return build_kernel()

def test_half_precision_input(cuda_module):
    # Issue 1: Kernel does not support half precision due to AT_DISPATCH_FLOATING_TYPES omission.
    # Create a half precision (float16) tensor. The extension is expected to throw an error.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # Call the kernel forward; it should fail or throw an error as half isn’t dispatched.
        y = cuda_module.forward(x)
        # Force device synchronization to catch asynchronous errors.
        torch.cuda.synchronize()

def test_non_contiguous_input(cuda_module):
    # Issue 2: The kernel assumes contiguous and properly aligned data.
    # Here we force a non-contiguous tensor by transposing a 2D tensor.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    # Create a non-contiguous tensor by transposing a tensor obtained from x.
    x_noncontig = x.t()  # Now size [16384, 16] and non-contiguous.
    # Although the kernel flattens using numel() and data_ptr(),
    # the expected result from PyTorch's hardsigmoid (which uses elementwise operations)
    # will not match the kernel outcome.
    y_kernel = cuda_module.forward(x_noncontig)
    # Compute expected output using PyTorch's functional implementation.
    y_expected = torch.nn.functional.hardsigmoid(x_noncontig)
    # Force device synchronization to ensure kernel completion.
    torch.cuda.synchronize()
    # Verify that the results are different (indicating that the non-contiguous input
    # is not correctly handled).
    # (In a robust implementation, the kernel would accept any tensor format.)
    assert not torch.allclose(y_kernel, y_expected, atol=1e-5), \
           "Kernel unexpectedly handled a non-contiguous input tensor correctly."

def test_missing_synchronization(cuda_module):
    # Issue 3: The kernel does not explicitly synchronize after launch.
    # We can attempt to trigger an error caused by asynchronous kernel failures by 
    # providing an input that is too large or misaligned through slicing.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    # Induce misalignment by slicing one element off (this can break the assumed 128-bit alignment)
    x_misaligned = x.reshape(-1)[1:].reshape(16, -1)
    # Run the kernel (it may launch without error immediately, but when synchronizing,
    # an error might be reported).
    y = cuda_module.forward(x_misaligned)
    # Forcing synchronization; if a runtime error occurs internally it should surface here.
    try:
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel launch error due to missing synchronization: {e}")
    # If no error is raised, then the test will at least inform that the lack of synchronization
    # may be hiding potential issues in more complex cases.
    # Additionally, compare with PyTorch's built-in hardsigmoid.
    y_expected = torch.nn.functional.hardsigmoid(x_misaligned)
    # In a correct implementation, the outputs would match; here we compare and if they are
    # different then it signals that the kernel did not correctly process the mis-aligned input.
    assert not torch.allclose(y, y_expected, atol=1e-5), \
           "Kernel output unexpectedly matches for misaligned input despite missing synchronization."
