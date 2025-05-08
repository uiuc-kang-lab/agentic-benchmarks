
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Helper function to build the kernel extension.
# We allow passing extra_cuda_cflags to modify BLOCK_SIZE and other macros.
def build_kernel(extra_cuda_cflags=None):
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

# Issue 1: Test launching the kernel with a block size NOT multiple of warpSize (e.g., 250)
def test_block_size_not_multiple_of_warpSize():
    # Override BLOCK_SIZE to 250 which is not divisible by 32.
    # This should cause wrong reduction or crash.
    # Note: The behavior is undefined so we simply check that the computed result
    # deviates significantly from the expected value.
    module = build_kernel(extra_cuda_cflags=["-O3", "--use_fast_math", "-DBLOCK_SIZE=250"])
    batch_size = 128
    num_elements = 4096
    shape = (num_elements,)
    # Use contiguous tensors here
    preds = torch.randn(batch_size, *shape, device="cuda", dtype=torch.float32)
    tgts = torch.randn(batch_size, *shape, device="cuda", dtype=torch.float32)
    result = module.forward(preds, tgts)
    # Compute reference
    ref = torch.mean((preds - tgts) ** 2)
    # With an incorrect reduction, the output is not expected to match.
    # We assert that the error is large.
    assert not torch.allclose(result, ref, atol=1e-5), (
        "Kernel with non-multiple BLOCK_SIZE produced result close to expected value, "
        "but it should be incorrect."
    )

# Issue 2: Test using a GPU that does not support atomicAdd on doubles.
def test_atomicAdd_double_support():
    # Check the compute capability of the current device.
    cap_major, cap_minor = torch.cuda.get_device_capability()
    # atomicAdd(double) is supported for compute capability >= 6.0.
    # If the device is older, we expect the kernel launch to fail.
    module = build_kernel()
    batch_size = 16
    shape = (512,)
    preds = torch.randn(batch_size, *shape, device="cuda", dtype=torch.float32)
    tgts = torch.randn(batch_size, *shape, device="cuda", dtype=torch.float32)
    if cap_major < 6:
        with pytest.raises(RuntimeError):
            # Expect a runtime error because atomicAdd on double is not supported.
            module.forward(preds, tgts)
    else:
        # For supported devices, the kernel should run and produce a correct numerical result.
        result = module.forward(preds, tgts)
        ref = torch.mean((preds - tgts) ** 2)
        assert torch.allclose(result, ref, atol=1e-5), (
            "Kernel output differs from reference on a device that supports atomicAdd for double."
        )

# Issue 3: Test with non-contiguous tensors.
def test_noncontiguous_input():
    module = build_kernel()
    batch_size = 64
    shape = (1024,)
    # Create base contiguous tensors
    base_preds = torch.randn(batch_size, *shape, device="cuda", dtype=torch.float32)
    base_tgts = torch.randn(batch_size, *shape, device="cuda", dtype=torch.float32)
    # Make non-contiguous versions via transpose or indexing.
    preds = base_preds.t().t()  # simple trick to get a non-contiguous view sometimes
    tgts = base_tgts.t().t()
    # Ensure they are not contiguous.
    assert not preds.is_contiguous(), "Predictions tensor should be non-contiguous for this test."
    # The kernel does not check for contiguity so it may produce an incorrect result.
    result = module.forward(preds, tgts)
    ref = torch.mean((preds - tgts) ** 2)
    # Expect the result to be different from the correct value.
    assert not torch.allclose(result, ref, atol=1e-5), (
        "Kernel did not exhibit an error with non-contiguous inputs even though it assumes contiguous memory."
    )

# Issue 4: Test with half-precision (float16) tensors.
def test_half_precision_input():
    module = build_kernel()
    batch_size = 32
    shape = (2048,)
    preds = torch.randn(batch_size, *shape, device="cuda", dtype=torch.float16)
    tgts = torch.randn(batch_size, *shape, device="cuda", dtype=torch.float16)
    # Since AT_DISPATCH_FLOATING_TYPES does not cover half, we expect an error.
    with pytest.raises(RuntimeError):
        module.forward(preds, tgts)
