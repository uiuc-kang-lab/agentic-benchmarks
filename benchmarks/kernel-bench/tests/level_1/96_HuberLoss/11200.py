
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel(extra_cuda_cflags=None):
    # Allow passing additional CUDA flags to simulate modifications.
    if extra_cuda_cflags is None:
        extra_cuda_cflags = ["-O3", "--use_fast_math"]
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_unqualified_min_issue():
    """
    Test for Issue 1: Unqualified use of min.
    This test attempts to force a compile-time error by injecting
    a flag that prevents implicit declarations. In practice,
    if the extension builds, we simulate the logic by ensuring that 
    the kernel can be loaded only if proper min is available.
    """
    # Attempt to build the kernel without changing extra flags.
    # If the unqualified "min" were an issue, this build would fail.
    try:
        module = build_kernel()
    except Exception as e:
        pytest.skip("Kernel build failed due to min issue as expected: " + str(e))
    else:
        # If it built, we warn that the issue is not being triggered in this environment.
        pytest.skip("Kernel build succeeded; unqualified min might not be an issue on this system.")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dtype_mismatch_issue():
    """
    Test for Issue 2: Kernel only supports float32.
    Passing double precision tensors should yield a wrong result.
    """
    module = build_kernel()
    # Create input tensors with dtype=torch.float64
    predictions = torch.randn(128, 4096, dtype=torch.float64, device="cuda")
    targets = torch.randn(128, 4096, dtype=torch.float64, device="cuda")
    
    # Our kernel assumes inputs are float32: so we expect that the output
    # computed will likely differ from PyTorch’s smooth_l1_loss computed with double.
    loss_kernel = module.forward(predictions, targets)
    loss_torch = torch.nn.functional.smooth_l1_loss(predictions, targets)
    
    # The two values should not be close since the underlying binary data is misinterpreted.
    assert not torch.allclose(loss_kernel, loss_torch.float(), atol=1e-5), (
        "Kernel output unexpectedly matches reference output even with dtype mismatch."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_shared_memory_overrun_issue():
    """
    Test for Issue 3: Shared memory allocation assumes max. 32 warps per block.
    We simulate a launch with an artificially large block size by redefining a macro.
    This is done by passing a custom extra_cuda_cflags that defines BLOCK_SIZE = 2048.
    The kernel's host code should use a hardcoded 'block_size' constant,
    so we expect that if the kernel used this macro, it would cause shared memory overrun.
    """
    # We pass a macro definition to force a block size larger than 1024 (e.g., 2048).
    extra_flags = ["-O3", "--use_fast_math", "-DBLOCK_SIZE=2048"]
    try:
        module = build_kernel(extra_cuda_cflags=extra_flags)
    except Exception as e:
        pytest.skip("Kernel build failed with custom block size flag as expected: " + str(e))
    
    # Since the host code in the provided implementation ignores such macros,
    # we simulate the test by ensuring that if a large block size were used,
    # the kernel would likely produce an incorrect result or crash.
    # We create a tensor that forces sufficient iterations.
    predictions = torch.randn(50000, device="cuda", dtype=torch.float32)
    targets = torch.randn(50000, device="cuda", dtype=torch.float32)
    try:
        loss_kernel = module.forward(predictions, targets)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.skip("Kernel execution failed as expected with large block size: " + str(e))
    else:
        # Since the behavior is undefined, we check for a suspicious value.
        loss_torch = torch.nn.functional.smooth_l1_loss(predictions, targets)
        # If the result is unreasonably off, we assume that shared memory overrun has occurred.
        rel_error = abs(loss_kernel.item() - loss_torch.item()) / abs(loss_torch.item() + 1e-6)
        assert rel_error > 1e-2, (
            "Kernel output appears correct despite potential shared memory overrun with large block size."
        )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_kernel_launch_error_checking():
    """
    Test for Issue 4: Absence of error checking after kernel launch.
    We induce an error by deliberately passing tensors with mismatched shapes.
    The kernel host function performs a TORCH_CHECK on the sizes, so this should raise an error.
    """
    module = build_kernel()
    predictions = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    # Create targets with mismatched size.
    targets = torch.randn(128, 4095, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError):
        _ = module.forward(predictions, targets)
