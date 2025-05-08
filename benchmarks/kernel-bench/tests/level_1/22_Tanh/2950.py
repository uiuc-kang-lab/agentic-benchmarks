
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

# Test case 1: Non-contiguous tensor input
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor then a non-contiguous view (by transposing)
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # Transpose makes it non-contiguous
    with pytest.raises(RuntimeError):
        # Expect a runtime error or incorrect behavior due to misinterpretation of memory layout.
        y = my_module.forward(x_noncontig)
        torch.cuda.synchronize()
        # The test forces an error by checking that the result is not valid.
        assert not y.is_contiguous(), "The kernel should not silently work on non-contiguous input."

# Test case 2: Misaligned memory (simulate by using a non-zero storage offset)
def test_misaligned_storage():
    my_module = build_kernel()
    # Create a tensor and then create another with an offset so that the underlying pointer may be misaligned.
    x = torch.randn(16, 16384 + 1, device="cuda", dtype=torch.float32)
    # Slice off the first element to force a non-zero storage offset.
    x_misaligned = x[:, 1:]
    if not x_misaligned.data_ptr() % 16 == 0:
        with pytest.raises(RuntimeError):
            y = my_module.forward(x_misaligned)
            torch.cuda.synchronize()
    else:
        pytest.skip("Tensor appears to be aligned, cannot trigger misalignment issue reliably.")

# Test case 3: Unsupported data type: half precision
def test_unsupported_dtype_half():
    my_module = build_kernel()
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # The extension dispatch does not support half precision, so it should raise an error.
        y = my_module.forward(x)
        torch.cuda.synchronize()

# Test case 4: Kernel launch error detection (simulate by using an extremely large size)
def test_large_tensor_launch_error():
    my_module = build_kernel()
    # This might trigger a kernel launch error if the limits are exceeded.
    # Note: Depending on the available GPU memory, this test might simply run very slowly.
    try:
        x = torch.randn(1, 2**28, device="cuda", dtype=torch.float32)  # Very large tensor
    except RuntimeError:
        pytest.skip("Skipping test_large_tensor_launch_error, not enough GPU memory.")
    # Force kernel launch. We expect that without error checking, kernel errors go unnoticed.
    y = my_module.forward(x)
    torch.cuda.synchronize()
    # Since no proper error checking is done within the kernel launch, we simply check that y has the expected shape.
    assert y.shape == x.shape, "Output shape is not as expected, possible kernel launch issues."

