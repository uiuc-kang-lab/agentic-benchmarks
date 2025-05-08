
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility: Load the CUDA extension module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Double precision error due to unconditionally using expf and float conversion.
def test_double_precision():
    kernel_module = build_kernel()
    # Create a tensor in double precision
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # Run our kernel forward: note that the kernel will compute in float precision.
    y_kernel = kernel_module.forward(x)
    # Compute reference with torch.sigmoid (keeps it in double)
    y_ref = torch.sigmoid(x)
    # The results will differ more than a tight tolerance because of precision loss.
    # We expect the relative difference to exceed a low tolerance.
    if torch.allclose(y_kernel, y_ref, atol=1e-8):
        pytest.fail("Kernel incorrectly handled double precision input without precision loss.")

# Issue 2: Potential indexing overflow when tensor has a huge number of elements.
@pytest.mark.skip(reason="Test for index overflow in extremely large tensors is not feasible in test environment.")
def test_large_tensor_indexing():
    kernel_module = build_kernel()
    # This test simulates a scenario where the number of elements exceeds the range of a 32-bit int.
    # Note: In a real scenario, one would allocate a huge tensor. For testing purposes, we simulate
    # a tensor whose numel() is patched to be a huge number.
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    # Monkey-patch the numel() method to simulate a huge tensor size.
    original_numel = x.numel()
    huge_size = (2 ** 31) + 1000  # beyond INT_MAX
    # We create a dummy tensor output of the same shape as x (contents don't matter).
    # Call the forward function and let it use the patched size.
    # Here we simulate by overriding the attribute in a wrapper.
    class FakeTensor(torch.Tensor):
        pass

    # Hack: set a fake numel attribute using a simple subclass.
    x_fake = x.clone().detach()
    x_fake.__class__ = FakeTensor
    x_fake.fake_numel = huge_size  # We'll assume our kernel uses this value (in real code it uses x.numel())

    # Since our actual forward computes size = x.numel(), we cannot inject huge_size without modifying the kernel.
    # So this test is only a placeholder to document the issue and is marked as skipped.
    pytest.skip("Simulated test for large tensor indexing is not feasible to run in the current environment.")

# Issue 3: Incorrect results for non-contiguous inputs.
def test_non_contiguous():
    kernel_module = build_kernel()
    # Create a contiguous tensor and then obtain a non-contiguous view
    x = torch.randn(1024, 16, device="cuda", dtype=torch.float32)
    x_t = x.t()  # transpose to make it non-contiguous
    if x_t.is_contiguous():
        pytest.skip("Test requires a non-contiguous tensor, but tensor is contiguous after transpose.")
    # Run kernel forward on non-contiguous tensor. The kernel doesn't check for contiguity.
    y_kernel = kernel_module.forward(x_t)
    # Compute reference result using torch.sigmoid on a contiguous copy.
    y_ref = torch.sigmoid(x_t.contiguous())
    # The outputs are expected to differ as the kernel assumes contiguous storage.
    if torch.allclose(y_kernel, y_ref, atol=1e-5):
        pytest.fail("Kernel output for non-contiguous input is unexpectedly correct. "
                    "Kernel should not work properly with non-contiguous inputs.")

if __name__ == "__main__":
    pytest.main([__file__])
