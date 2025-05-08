
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="mse_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

###############################################################################
# Test 1: Trigger issue with half precision inputs.
###############################################################################
def test_half_precision_input():
    # Create half precision tensors on CUDA.
    # Expect the kernel to throw an error because half is not dispatched.
    cuda_module = build_kernel()
    preds = torch.randn(128, 4096, device='cuda', dtype=torch.float16)
    tgts = torch.randn(128, 4096, device='cuda', dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # Should fail because AT_DISPATCH_FLOATING_TYPES does not include half.
        out = cuda_module.forward(preds, tgts)
        torch.cuda.synchronize()

###############################################################################
# Test 2: Trigger issue with empty tensors.
###############################################################################
def test_empty_tensor():
    cuda_module = build_kernel()
    # Create empty but CUDA tensors.
    preds = torch.empty((0,), device='cuda', dtype=torch.float32)
    tgts = torch.empty((0,), device='cuda', dtype=torch.float32)
    # Depending on the behavior, division by zero may happen.
    # We either expect an error or a NaN result.
    out = cuda_module.forward(preds, tgts)
    torch.cuda.synchronize()
    # If no exception, the output should be NaN (or some undefined value).
    assert torch.isnan(out).item(), "Expected NaN due to division by zero on empty tensor"

###############################################################################
# Test 3: Trigger issue with non-contiguous input tensors.
###############################################################################
def test_non_contiguous_input():
    cuda_module = build_kernel()
    # Create contiguous tensors first.
    preds = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    tgts = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    # Make them non-contiguous (e.g., by transposing a 2D tensor).
    preds_nc = preds.t()
    tgts_nc = tgts.t()
    # Compute kernel output and reference output using PyTorch.
    out_kernel = cuda_module.forward(preds_nc, tgts_nc)
    torch.cuda.synchronize()
    out_ref = torch.mean((preds_nc - tgts_nc) ** 2)
    # The kernel assumes contiguous memory so its result will likely differ.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-4), "Kernel output unexpectedly matches reference for non-contiguous tensors"

###############################################################################
# Test 4: Trigger issue with atomicAdd(double) on GPUs with insufficient support.
###############################################################################
def test_atomicAdd_double_support():
    cuda_module = build_kernel()
    # Check device compute capability.
    cc = torch.cuda.get_device_capability()
    preds = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    tgts = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    if cc[0] < 6:
        with pytest.raises(RuntimeError):
            # The kernel should fail on devices with compute capability < 6.0
            out = cuda_module.forward(preds, tgts)
            torch.cuda.synchronize()
    else:
        # On supported devices, the kernel should run without error.
        out = cuda_module.forward(preds, tgts)
        torch.cuda.synchronize()
        # Verify that the result is within a reasonable range.
        out_ref = torch.mean((preds - tgts) ** 2)
        assert torch.allclose(out, out_ref, atol=1e-5), "Kernel output differs from reference on supported device"

###############################################################################
# Test 5: Trigger issue with missing CUDA error checking after kernel launch.
###############################################################################
def test_missing_cuda_error_check():
    cuda_module = build_kernel()
    # Provide CPU tensors to trigger the device check in the forward function.
    preds_cpu = torch.randn(128, 4096, device="cpu", dtype=torch.float32)
    tgts_cpu = torch.randn(128, 4096, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(preds_cpu, tgts_cpu)
        torch.cuda.synchronize()
