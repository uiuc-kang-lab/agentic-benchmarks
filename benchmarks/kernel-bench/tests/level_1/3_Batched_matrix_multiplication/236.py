
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the CUDA kernel from "kernel.cu"
def build_kernel():
    cuda_module = load(
        name="bmm_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function: reference batched matrix multiplication
def ref_bmm(A, B):
    return torch.bmm(A, B)

# Issue 1: Kernel only supports float32, so passing non-float32 (e.g. double) should trigger an error or produce incorrect values.
def test_input_tensor_type():
    cuda_module = build_kernel()
    batch_size, m, k, n = 4, 16, 32, 20
    # Create double precision tensors
    A = torch.randn(batch_size, m, k, device="cuda", dtype=torch.double)
    B = torch.randn(batch_size, k, n, device="cuda", dtype=torch.double)
    
    # The kernel expects float; since we are directly passing A.data_ptr<float>(), an incorrect type is used.
    # Depending on backend conversions, we may get incorrect results.
    # We therefore check that the CUDA kernel output does NOT match the reference result computed in double precision.
    C_cuda = cuda_module.forward(A.float(), B.float())  # correct type call for control
    C_ref = ref_bmm(A.float(), B.float())
    assert torch.allclose(C_cuda, C_ref, atol=1e-5), "Kernel failed on float32 inputs"

    # Now intentionally pass double tensors (via reinterpretation): we use .double() and then cast the pointer.
    # This is unsafe; we expect the results to be erroneous.
    # In a proper implementation, the kernel would check for the dtype and raise an error.
    with pytest.raises(RuntimeError):
        # This call should fail since the underlying data is not float and the kernel is not templated.
        _ = cuda_module.forward(A, B)

# Issue 2: Kernel assumes contiguous memory. If non-contiguous tensors are provided the result may be incorrect.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch_size, m, k, n = 4, 16, 24, 20  # chosen dimensions (k not a multiple of TILE_SIZE does not matter here)
    # Create contiguous tensors
    A = torch.randn(batch_size, m, k, device="cuda", dtype=torch.float32)
    B = torch.randn(batch_size, k, n, device="cuda", dtype=torch.float32)
    # Make non-contiguous copies by transposing inner dimensions then transposing back.
    A_noncontig = A.transpose(1, 2).transpose(1, 2)
    B_noncontig = B.transpose(1, 2).transpose(1, 2)
    assert not A_noncontig.is_contiguous(), "A should be non-contiguous"
    assert not B_noncontig.is_contiguous(), "B should be non-contiguous"
    
    # Run kernel on non-contiguous tensors (expect incorrect result)
    C_cuda = cuda_module.forward(A_noncontig, B_noncontig)
    C_ref = ref_bmm(A_noncontig.contiguous(), B_noncontig.contiguous())
    # We expect the results to differ since the kernel does not handle non-contiguous inputs.
    if torch.allclose(C_cuda, C_ref, atol=1e-5):
        pytest.fail("Kernel unexpectedly handled non-contiguous inputs correctly; expected failure.")

# Issue 3: The shared memory loading for matrix B may lead to bank conflicts,
# especially when matrix dimensions are not multiples of TILE_SIZE.
# Although the kernel may produce correct results, its performance can suffer.
# We design a test case with dimensions that are not divisible by TILE_SIZE.
def test_non_divisible_dims():
    cuda_module = build_kernel()
    # use sizes that are not multiples of TILE_SIZE (32)
    batch_size, m, k, n = 4, 37, 50, 45
    A = torch.randn(batch_size, m, k, device="cuda", dtype=torch.float32)
    B = torch.randn(batch_size, k, n, device="cuda", dtype=torch.float32)
    
    # Run the kernel and reference bmm
    C_cuda = cuda_module.forward(A, B)
    C_ref = ref_bmm(A, B)
    # Correctness should be maintained but note that performance may be degraded in these cases.
    assert torch.allclose(C_cuda, C_ref, atol=1e-5), "Kernel output is incorrect for non-divisible dimensions"

