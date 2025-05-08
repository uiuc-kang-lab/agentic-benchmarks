
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dtype_mismatch():
    """
    Test case for Issue #1:
    Pass inputs with a dtype other than float32.
    The kernel expects float32 but we deliberately call it with float64.
    The output of the kernel is expected to be incorrect relative to the PyTorch result.
    """
    N = 1024
    M = 1024
    # Create inputs with float64 dtype on CUDA.
    A = torch.randn(N, dtype=torch.float64, device="cuda")
    B = torch.randn(N, M, dtype=torch.float64, device="cuda")
    module = build_kernel()
    # Call the CUDA kernel (which uses float pointers) with float64 tensors.
    C = module.forward(A, B)
    torch.cuda.synchronize()
    # Compute expected result using PyTorch. Since the kernel is using reinterpretation,
    # the output will not be close to the reference.
    C_ref = torch.diag(A) @ B
    # It is very likely that the output is NOT close to the expected result due to incorrect interpretation.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Kernel output unexpectedly matches reference output for float64 input; "
        "data type mismatch issue may not be present."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_integer_overflow():
    """
    Test case for Issue #2:
    Provide inputs with very large dimensions such that N*M exceeds the max value of a 32-bit int.
    This should expose the potential integer overflow in the kernel's index computations.
    To avoid allocating an unrealistic amount of memory, we check available memory first.

    Note: For many GPUs it may be impossible to allocate a tensor of the oversized dimensions,
    so this test will be skipped if there isn't enough available memory.
    """
    # Choose dimensions such that total elements > 2^31 (approx 2.147e9)
    # Here we choose N=M=50000; total elements = 2.5e9, which would be 2.5e9*4 bytes = ~10GB
    # Check if the GPU has enough memory (this check is approximate)
    props = torch.cuda.get_device_properties(0)
    available_memory = torch.cuda.memory_reserved(0)  # current reserved memory in bytes
    total_memory = props.total_memory
    required_bytes = 50000 * 50000 * 4  # float32 bytes
    if required_bytes > total_memory // 2:
        pytest.skip("Skipping test_integer_overflow due to insufficient available GPU memory.")
    
    N = 50000
    M = 50000
    # Allocate inputs.
    A = torch.rand(N, dtype=torch.float32, device="cuda")
    B = torch.rand(N, M, dtype=torch.float32, device="cuda")
    module = build_kernel()
    # Run the kernel.
    C = module.forward(A, B)
    torch.cuda.synchronize()

    # Verify a few random elements.
    # With correct execution, C[i,j] should be equal to A[i]*B[i,j].
    # Due to integer overflow in the kernel’s loop index calculations, we expect discrepancies.
    # Here we test a few elements.
    idxs_to_test = [0, N * M // 2, N * M - 1]
    for idx in idxs_to_test:
        row = idx // M
        # In the correctly computed result, we expect:
        expected = A[row].item() * B.view(-1)[idx].item()
        actual = C.view(-1)[idx].item()
        assert abs(actual - expected) > 1e-3, (
            f"At flat index {idx}, expected a large difference due to overflow, "
            f"but got actual={actual} and expected={expected}"
        )
