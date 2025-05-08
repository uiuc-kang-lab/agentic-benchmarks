
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="custom_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Kernel only supports float32 inputs.
def test_non_float32_input():
    module = build_kernel()
    # Create a tensor with dtype=torch.float64.
    A = torch.randn(1024, 1024, dtype=torch.float64, device="cuda")
    s = 3.14
    with pytest.raises(RuntimeError, match="Input must be float32"):
        # This should trigger TORCH_CHECK that tests the scalar type.
        _ = module.forward(A, s)

# Issue 2: Kernel assumes that the input tensor is contiguous and properly aligned.
def test_non_contiguous_input():
    module = build_kernel()
    # Create a contiguous tensor and then make a non-contiguous slice.
    A = torch.randn(128, 128, dtype=torch.float32, device="cuda")
    # Transpose returns a non-contiguous tensor.
    A_non_contig = A.t()
    s = 2.5
    with pytest.raises(RuntimeError, match="Input tensor must be contiguous"):
        _ = module.forward(A_non_contig, s)

# Issue 3: Kernel assumes 16-byte alignment due to vectorized (float4) loads.
def test_misaligned_input():
    module = build_kernel()
    # We force misalignment by creating a 1D tensor with extra element,
    # then slicing off the first element so that the data_ptr is offset.
    M = 64
    N = 64
    total_elems = M * N + 1
    base = torch.randn(total_elems, dtype=torch.float32, device="cuda")
    # Create a misaligned tensor view by skipping the first element.
    A = base.narrow(0, 1, M * N).reshape(M, N).contiguous()
    s = 1.5
    # Even though A is contiguous, its underlying data_ptr is now misaligned.
    # The kernel’s reinterpret_cast to float4 may produce incorrect results.
    C = module.forward(A, s)
    # Compute the expected result on CPU (or using PyTorch mul)
    C_expected = A * s
    # We expect the kernel result NOT to match perfectly due to misalignment issues.
    # (On some hardware misaligned accesses don't crash but lead to incorrect performance or values.)
    # Here, we trigger the issue by testing that the result is NOT close to the reference.
    assert not torch.allclose(C, C_expected, atol=1e-6), "Kernel unexpectedly handled misaligned input correctly."

# Issue 4: No error checking after kernel launch.
def test_kernel_launch_error_checking():
    module = build_kernel()
    # Force a kernel launch error by passing an empty tensor that would lead to total_elements==0.
    A = torch.empty(0, dtype=torch.float32, device="cuda")
    s = 3.14
    # While the empty tensor case might not produce a launch error in many situations,
    # we can simulate a failure by expecting some abnormal behavior.
    with pytest.raises(RuntimeError):
        _ = module.forward(A, s)
