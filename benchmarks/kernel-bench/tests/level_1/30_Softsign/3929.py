
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension kernel.
def build_kernel(extra_cuda_flags=None):
    # Using a unique name each time to force a recompilation if needed.
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_flags if extra_cuda_flags is not None else ["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Wrong dtype handling
def test_dtype_incompatibility():
    """
    The kernel always treats the input as float32. Passing a tensor with a different dtype (e.g. float64)
    will lead to misinterpreted memory. In this test we check that the kernel output does NOT match
    the expected softsign computed by PyTorch.
    """
    kernel_module = build_kernel()
    x = torch.randn(1024, device='cuda', dtype=torch.double)
    # Expected softsign computed with full precision (double)
    expected = x / (1 + torch.abs(x))
    # Kernel will reinterpret the double tensor data as float32 values.
    out = kernel_module.forward(x)
    # The output should be very different from expected (or at least not close).
    assert not torch.allclose(out.double(), expected, atol=1e-5), \
        "Kernel incorrectly handled non-float32 input; outputs match even though the dtype is wrong."


# Issue 2: Missing error check after kernel launch
def test_kernel_launch_error_checking():
    """
    Without an error check after launching the kernel (e.g. calling cudaGetLastError),
    a misconfigured launch might go unnoticed. One way to trigger a launch failure is by trying
    to launch the kernel with an enormous input that should exceed available resources.
    (Note: This test may be hardware dependent.)
    """
    # Try to allocate a huge tensor (e.g. ~1e9 elements ~ 4GB) so that the kernel launch is likely to fail.
    huge_elems = 10**9
    try:
        x = torch.randn(huge_elems, device='cuda', dtype=torch.float32)
    except RuntimeError:
        pytest.skip("Skipping test_kernel_launch_error_checking because huge tensor allocation failed on this device.")

    kernel_module = build_kernel()
    # We expect that the kernel launch may fail.
    with pytest.raises(RuntimeError):
        out = kernel_module.forward(x)
        # Force synchronization to ensure kernel errors show up.
        torch.cuda.synchronize()


# Issue 3: Unqualified use of min
def test_compilation_min_issue():
    """
    The kernel code uses "min(blocks, max_blocks)" without proper qualification.
    This should normally cause a compile error unless min is provided by some other header.
    In our test, we force a rebuild of the extension and expect a compile-time failure.
    """
    # We force a rebuild by using an extra flag (which should not affect the logic).
    # The compilation is expected to fail, so load() should raise an Exception.
    with pytest.raises(Exception) as excinfo:
        build_kernel(extra_cuda_flags=["-O3", "--use_fast_math", "-Xcompiler", "-Werror"])
    err_msg = str(excinfo.value)
    assert "min" in err_msg or "std::min" in err_msg, \
        "Expected compile error due to unqualified use of min was not found."

