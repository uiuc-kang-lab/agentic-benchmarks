
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Test that passing a non-float32 tensor (e.g. float64) results in an error or produces incorrect outputs.
def test_input_tensor_type():
    cuda_module = build_kernel()
    N = 128
    M = 64
    # Create A as a 1D tensor of type float64 and B as a 2D tensor of type float64 on CUDA.
    A = torch.randn(N, dtype=torch.double, device='cuda')
    B = torch.randn(N, M, dtype=torch.double, device='cuda')
    # The kernel expects float32 inputs. Running with float64 may produce wrong results or trigger a runtime error.
    with pytest.raises(RuntimeError):
        # If the kernel misinterprets the data, synchronization or result retrieval should fail.
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Test that the fixed grid configuration (<<<N, 256>>>) causes issues in more general/large shape situations.
def test_large_grid_configuration():
    cuda_module = build_kernel()
    # Choose a large N to force a huge number of blocks.
    # Note: On some GPUs this might work, but on others the launch configuration may exceed the hardware limits.
    # We use a try/except and expect a runtime error on devices with capped grid dimensions.
    N = 50000  # Large number of rows
    M = 32
    A = torch.randn(N, dtype=torch.float32, device='cuda')
    B = torch.randn(N, M, dtype=torch.float32, device='cuda')
    try:
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()
    except RuntimeError as e:
        pytest.skip("GPU grid dimension exceeded: " + str(e))
    # If it does run, verify that the multiplication is performed correctly.
    # The operation is C[i,j] = A[i] * B[i,j]
    C_ref = A.unsqueeze(1) * B
    assert torch.allclose(C, C_ref, atol=1e-5), "Kernel with large grid configuration produced incorrect result."

# Issue 3 & 4 are about naming/documentation and missing error checks.
# It is difficult to trigger a runtime error from missing error checking unless the kernel launch genuinely fails,
# so we include a basic test using correct inputs to illustrate that the kernel run appears successful.
def test_correct_output_on_normal_input():
    cuda_module = build_kernel()
    N = 1024
    M = 512
    A = torch.randn(N, dtype=torch.float32, device='cuda')
    B = torch.randn(N, M, dtype=torch.float32, device='cuda')
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # Expected result: C[i,j] = A[i] * B[i,j]
    C_ref = A.unsqueeze(1) * B
    assert torch.allclose(C, C_ref, atol=1e-5), "Kernel produced incorrect result on normal input."
