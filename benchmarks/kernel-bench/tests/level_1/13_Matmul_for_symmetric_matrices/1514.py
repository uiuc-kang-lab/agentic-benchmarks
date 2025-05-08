
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

# Issue 1: Kernel only supports square matrices.
def test_non_square_matrices():
    my_module = build_kernel()
    # Create non-square matrices. The forward function performs checks for square matrices.
    A = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    B = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Should trigger an error from the TORCH_CHECK in forward because A and B are not square.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Kernel supports only float32 inputs.
def test_double_dtype_inputs():
    my_module = build_kernel()
    # Create square symmetric matrices with dtype float64.
    N = 256
    A = torch.randn(N, N, device="cuda", dtype=torch.float64)
    A = (A + A.t()) / 2
    B = torch.randn(N, N, device="cuda", dtype=torch.float64)
    B = (B + B.t()) / 2
    
    # The forward function does not check dtype so the kernel will be launched with A and B interpreted as float,
    # leading to an incorrect result.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result with proper type conversion to float32.
    C_ref = torch.matmul(A.float(), B.float())
    # The result is expected to be wrong because of the misinterpretation of the memory.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel unexpectedly produced correct results for double inputs, but it should not."

# Issue 3: Lack of error checking for CUDA kernel launch errors.
# We simulate a situation that tends to trigger a kernel launch configuration error.
def test_kernel_launch_error():
    my_module = build_kernel()
    # Create inputs that are square and float32 but set an invalid size by forcing a very large dimension 
    # (likely to exceed GPU limits for grid dimensions).
    # Note: The forward function infers N from A.size(0) and expects a square matrix so we force an absurd size.
    N = 1 << 16  # 65536; may cause grid dimension issues on many GPUs.
    try:
        A = torch.randn(N, N, device="cuda", dtype=torch.float32)
        # Make symmetric
        A = (A + A.t()) / 2
        B = torch.randn(N, N, device="cuda", dtype=torch.float32)
        B = (B + B.t()) / 2
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
        # If no error was thrown but grid configuration is invalid, the results might be incorrect.
        C_ref = torch.matmul(A, B)
        # We expect the result to be off.
        assert not torch.allclose(C, C_ref, atol=1e-3), \
            "Kernel launch error was expected due to invalid grid dimensions, but the result is unexpectedly close."
    except RuntimeError as e:
        # An error thrown here indicates that the kernel launch error is not properly caught inside the kernel API.
        # The test passes if a RuntimeError is raised.
        pytest.skip("Kernel launch failure detected and propagated: " + str(e))
