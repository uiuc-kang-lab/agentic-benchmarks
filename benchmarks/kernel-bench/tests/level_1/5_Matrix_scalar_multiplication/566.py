
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension module from kernel.cu.
# Assume that kernel.cu is in the same directory as this test file.
def build_kernel():
    module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# 1. Test case for non-divisible-by-4 tensor size.
#    This should trigger an issue because the kernel assumes total size is a multiple of 4.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_divisible_by_four():
    my_module = build_kernel()
    # Create a tensor with 9 elements (e.g., shape (3, 3)), not divisible by 4.
    A = torch.randn(3, 3, device="cuda", dtype=torch.float32)
    s = 3.14
    # Depending on how the kernel is invoked, it might silently drop the tail elements
    # or lead to out-of-bound access. Here we run the kernel and then compare with reference.
    C = my_module.forward(A, s)
    # Compute reference using PyTorch:
    C_ref = A * s
    # Since the kernel’s behavior on the last few elements is undefined, we require that
    # the kernel result is not exactly correct. In a proper test, we are checking that the
    # issue is triggered.
    if torch.allclose(C, C_ref, atol=1e-5):
        pytest.fail("Test for non-divisible-by-4 size did not trigger the suspected issue.")

# 2. Test case for non-float32 input.
#    The kernel explicitly checks for float tensors so this should raise an error.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_float32_input():
    my_module = build_kernel()
    # Create a double precision tensor which should trigger the TORCH_CHECK
    A = torch.randn(1024, 1024, device="cuda", dtype=torch.double)
    s = 2.71
    with pytest.raises(RuntimeError):
        # The TORCH_CHECK should trigger an error indicating wrong data type.
        my_module.forward(A, s)

# 3. Test case for non-contiguous tensor.
#    Create a non-contiguous tensor (e.g. by transposing) to trigger potential misaligned memory access.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor and then transpose it so that it becomes non-contiguous.
    A = torch.randn(64, 128, device="cuda", dtype=torch.float32).t()  # non-contiguous after transpose
    s = 1.23
    # Although the TORCH_CHECK does not check contiguity, using reinterpret_cast on a non-contiguous tensor
    # may produce incorrect results.
    C = my_module.forward(A, s)
    C_ref = A * s
    if torch.allclose(C, C_ref, atol=1e-5):
        pytest.fail("Test for non-contiguous input did not trigger the suspected misaligned/vectorization issue.")

# 4. Test case to "detect" unused warp variables.
#    While this is not a runtime error, we try to trigger a warning by running on a tensor that relies on
#    potential warp optimizations. In this minimal kernel the warp_id and lane_id are not used,
#    and we provide a dummy test that simply runs the kernel on a larger input.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_tensor_for_warp_optimization():
    my_module = build_kernel()
    # Create a large tensor that forces the kernel to span many warps.
    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    s = 0.5
    C = my_module.forward(A, s)
    C_ref = A * s
    # Even though the result may numerically match,
    # the presence of unused warp_id and lane_id variables indicates unfulfilled potential optimizations.
    if not torch.allclose(C, C_ref, atol=1e-5):
        pytest.fail("Kernel output mismatch on large tensor indicating a potential error in warp usage.")
