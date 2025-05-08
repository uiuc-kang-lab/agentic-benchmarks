
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    # Assuming kernel.cu is in the same directory as this test file.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_module = load(
        name="hardtanh_cuda_module",
        sources=[os.path.join(this_dir, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_kernel():
    return build_kernel()

# Test case 1: Trigger misaligned access by creating a tensor view with a storage offset.
def test_misaligned_input(cuda_kernel):
    # Create a base tensor with extra elements to allow a storage offset.
    # The base tensor is likely 16-byte aligned. By narrowing with offset 1,
    # we force the returned view to reference memory starting at a misaligned address.
    base = torch.randn(17, device='cuda', dtype=torch.float32)
    # Create a view that starts at offset 1 and has length 16 (divisible by 4)
    x = base.narrow(0, 1, 16)
    # x is contiguous? .narrow returns a contiguous view only if there is no stride change.
    # However, even if contiguous, its storage offset makes its data_ptr misaligned.
    # Call the kernel with such input.
    out = cuda_kernel.forward(x, -1.0, 1.0)
    # Since the kernel assumes proper alignment for vectorized access,
    # the output may be incorrect. We trigger the issue by checking if the output
    # does not match the CPU PyTorch implementation.
    expected = torch.nn.functional.hardtanh(x, min_val=-1.0, max_val=1.0)
    # It is possible that the misaligned access may not crash but produce a wrong result.
    # We use an assertion that will fail if the kernel gives the "correct" result.
    # (In real debugging, one would check for runtime errors or use profiler tools.)
    if torch.allclose(out, expected):
        pytest.fail("Kernel did not produce a misalignment error on misaligned input.")

# Test case 2: Trigger incorrect behavior with non-contiguous input.
def test_non_contiguous_input(cuda_kernel):
    # Create a 2D tensor and then transpose it to make it non-contiguous.
    x = torch.randn(64, 64, device='cuda', dtype=torch.float32)
    x_t = x.t()  # transpose produces a non-contiguous tensor.
    # Ensure total number of elements is divisible by 4, so the vectorized kernel is chosen.
    assert x_t.numel() % 4 == 0, "Test requires numel divisible by 4."

    out = cuda_kernel.forward(x_t, -1.0, 1.0)
    expected = torch.nn.functional.hardtanh(x_t, min_val=-1.0, max_val=1.0)
    # The kernel ignores strides and treats the input as contiguous, so the result is expected to be wrong.
    if torch.allclose(out, expected, atol=1e-5):
        pytest.fail("Kernel produced correct result on non-contiguous input even though it assumes contiguity.")

# Test case 3: Trigger unsupported half precision.
def test_half_precision_input(cuda_kernel):
    # Create a half precision tensor.
    x = torch.randn(1024, device='cuda', dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # Expect a type dispatch error because half is not handled by AT_DISPATCH_FLOATING_TYPES
        cuda_kernel.forward(x, -1.0, 1.0)
