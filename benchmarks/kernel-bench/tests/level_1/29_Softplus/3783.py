
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="softplus_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Unsupport type (e.g. half precision or int) should trigger an error.
def test_unsupported_tensor_type():
    my_module = build_kernel()
    # Create a half tensor (float16) on CUDA, which is unsupported.
    x_half = torch.randn(1024, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # The kernel dispatch via AT_DISPATCH_FLOATING_TYPES will not match half tensor.
        my_module.forward(x_half)

    # Test for integer type tensor.
    x_int = torch.randint(0, 100, (1024,), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        my_module.forward(x_int)

# Issue 2: No check for CUDA device input; passing a CPU tensor should error.
def test_cpu_tensor_input():
    my_module = build_kernel()
    x_cpu = torch.randn(1024, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        my_module.forward(x_cpu)

# Issue 3: Kernel assumes contiguous memory; noncontiguous tensors may produce wrong results.
def test_noncontiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous by transposing
    x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # Transpose makes it non-contiguous in PyTorch
    assert not x_noncontig.is_contiguous(), "Tensor is unexpectedly contiguous."
    # The kernel uses x.data_ptr() assuming flat contiguous memory.
    # We do not have a ground truth here, but we expect the results to be incorrect.
    # One way to test is to compare with PyTorch's softplus and assert they differ.
    output_kernel = my_module.forward(x_noncontig)
    output_reference = torch.nn.functional.softplus(x_noncontig)
    # Since memory layout is not as expected, the kernel result may not match the reference.
    # We trigger the failure by asserting that they are not close.
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(output_kernel, output_reference, atol=1e-4)

# Issue 4: No error checking after kernel launch. We simulate a bad launch by forcing an invalid launch configuration.
# One way to trigger an error is to call the kernel with an empty tensor or an absurd number of threads.
def test_kernel_launch_error():
    my_module = build_kernel()
    # Create a dummy tensor with valid type and device.
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    # Force an invalid launch configuration by monkey-patching the kernel function's grid dimensions.
    # For that, we will pass an input size that is negative.
    # As we cannot easily change the launch parameters from Python, we simulate an error by calling the function
    # with a corrupted tensor shape by selecting a zero element tensor.
    x_empty = torch.empty(0, device="cuda", dtype=torch.float32)
    try:
        # Even though an empty tensor is often allowed, let’s assume the lack of synchronization will hide errors.
        # So we call forward and manually force synchronization.
        out = my_module.forward(x_empty)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.skip("Kernel launch error detected as expected: " + str(e))
    else:
        # If no error, we force a failure to indicate that error checking is missing.
        pytest.fail("Kernel did not report an error on launch with improper configuration.")
