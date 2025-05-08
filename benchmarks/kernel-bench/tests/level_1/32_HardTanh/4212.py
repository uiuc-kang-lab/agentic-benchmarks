
import os
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension module.
def build_kernel():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    module = load(
        name="hardtanh_cuda_module",
        sources=[os.path.join(this_dir, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Test for wrong input tensor type (non-float32)
def test_incorrect_dtype():
    # Create a tensor using double precision.
    x = torch.randn(256, device="cuda", dtype=torch.float64)
    module = build_kernel()
    # The kernel was compiled assuming float pointers.
    # If the kernel is used with a double tensor, the results will be incorrect.
    out = module.forward(x, -1.0, 1.0)
    # Use PyTorch's built-in hardtanh for reference, which works for double.
    expected = torch.clamp(x, min=-1.0, max=1.0)
    # We expect the kernel to not produce the same results because it treated the data as float.
    # Therefore, the test passes if the outputs differ significantly.
    assert not torch.allclose(out.to(torch.float64), expected, atol=1e-5), \
        "Kernel unexpectedly produced correct results for a non-float32 tensor!"

# Issue 2: Test for non-contiguous input tensor.
def test_noncontiguous_tensor():
    # Create a contiguous tensor and then create a noncontiguous view.
    x = torch.randn(32, 8, device="cuda", dtype=torch.float32)
    # Transpose to make it noncontiguous.
    x_noncont = x.t()
    assert not x_noncont.is_contiguous(), "x_noncont is unexpectedly contiguous."

    module = build_kernel()
    out = module.forward(x_noncont, -1.0, 1.0)
    # Use PyTorch's built-in clamp (hardtanh) for reference.
    expected = torch.clamp(x_noncont, min=-1.0, max=1.0)
    # Due to noncontiguity, the kernel’s direct pointer arithmetic may produce wrong results.
    # We expect the result not to match the reference.
    assert not torch.allclose(out, expected, atol=1e-5), \
        "Kernel unexpectedly produced correct results for a noncontiguous tensor!"

# Issue 3: Test that indirectly verifies that unused warp/lane computations have no side effects.
# Although the unused warp and lane variables do not change functionality,
# we provide a dummy test that runs the kernel and checks for basic correctness
# on a typical input. (This test will pass even with the unused variables.)
def test_basic_functionality():
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    module = build_kernel()
    out = module.forward(x, -1.0, 1.0)
    expected = torch.clamp(x, min=-1.0, max=1.0)
    # Because this test uses contiguous float32 tensors, it should work correctly.
    assert torch.allclose(out, expected, atol=1e-5), \
        "Kernel failed basic functionality test even on contiguous float32 tensor."
        
if __name__ == "__main__":
    pytest.main([__file__])
