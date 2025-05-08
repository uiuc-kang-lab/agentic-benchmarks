
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="mse_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# Test case 1: Non-contiguous input tensors.
def test_non_contiguous_tensors():
    # Create contiguous tensors and then make them non-contiguous by a view operation.
    A = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    B = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    # Make them non-contiguous by transposing (or slicing with a step)
    A_nc = A.t()
    B_nc = B.t()
    mse_kernel = build_kernel()
    # The kernel expects contiguous memory so it may work incorrectly.
    # We trigger a failure if the result is not as expected.
    result = mse_kernel.forward(A_nc, B_nc)
    # The expected result computed using PyTorch's mse loss
    expected = torch.mean((A_nc - B_nc) ** 2)
    torch.cuda.synchronize()
    assert not torch.allclose(result.cpu(), expected.cpu(), atol=1e-5), \
        "Kernel did not detect non-contiguous tensors as an issue."

# Test case 2: Index integer overflow for large tensors.
def test_large_tensor_index_overflow():
    # Note: Allocating >2^31 elements is usually not feasible.
    # Instead, we simulate the potential for overflow by using a moderately large tensor.
    # If the indexing is done with 32-bit ints, even moderately large tensor dimensions might trigger errors.
    N = 300000  # This number is modest but may expose issues in indexing logic.
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    mse_kernel = build_kernel()
    result = mse_kernel.forward(A, B)
    expected = torch.mean((A - B) ** 2)
    torch.cuda.synchronize()
    # If indexing overflows then the result would be incorrect.
    assert not torch.allclose(result.cpu(), expected.cpu(), atol=1e-5), \
        "Kernel indexing may be using 32-bit ints causing overflow issues on large tensors."

# Test case 3: Reduction unrolling assumption when blockDim.x < 32.
def test_reduction_unrolling():
    # We simulate a scenario where the number of elements is less than 32 (forcing blockDim.x to be effectively less than 32)
    # Although BLOCK_SIZE is hardcoded, using a very small tensor forces some threads to be inactive.
    N = 16  # Less than 32 elements.
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    mse_kernel = build_kernel()
    result = mse_kernel.forward(A, B)
    expected = torch.mean((A - B) ** 2)
    torch.cuda.synchronize()
    # If reduction unrolling fails for small block sizes, the result will be off.
    assert not torch.allclose(result.cpu(), expected.cpu(), atol=1e-5), \
        "Kernel reduction unrolling appears incorrect for blockDim.x < 32."

# Test case 4: AtomicAdd on double not supported on older architectures.
def test_atomicAdd_double_support():
    # This test does not change the GPU hardware but attempts to catch whether a runtime error occurs
    # because atomicAdd for double is called on an unsupported architecture.
    N = 1024
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    mse_kernel = build_kernel()
    try:
        result = mse_kernel.forward(A, B)
        torch.cuda.synchronize()
    except RuntimeError as e:
        err_str = str(e)
        assert "atomicAdd" in err_str, "Error does not mention atomicAdd on double."
    else:
        # If no error is raised, we assume that the hardware supports atomicAdd on double.
        pytest.skip("AtomicAdd on double is supported on this GPU architecture.")
