
import torch
import pytest
import threading
from torch.utils.cpp_extension import load

# Helper function to build/load our CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Constant memory limitation.
# We create a weight tensor whose number of elements exceeds 16K (16384 floats).
def test_constant_memory_limit():
    module = build_kernel()
    # Create a trivial input tensor.
    x = torch.randn(1, 1, 10, 10, device="cuda", dtype=torch.float32)
    # Create a weight tensor with shape (1, 1, 129, 129): 1*1*129*129 = 16641 > 16384.
    weight = torch.randn(1, 1, 129, 129, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Weight tensor too large for constant memory"):
        module.forward(x, weight, None, 1, 0, 1, 1)

# Issue 2: Lack of dilation support in kernel.
# When dilation != 1, the custom kernel does not handle it so the call falls back to torch::conv2d.
# This test verifies that behavior by comparing to the PyTorch implementation.
def test_dilation_support():
    module = build_kernel()
    x = torch.randn(1, 1, 10, 10, device="cuda", dtype=torch.float32)
    weight = torch.randn(1, 1, 3, 3, device="cuda", dtype=torch.float32)
    dilation = 2  # non-default dilation
    # The forward function falls back to torch conv2d.
    out_custom = module.forward(x, weight, None, 1, 1, dilation, 1)
    out_ref = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=1, dilation=dilation, groups=1)
    assert torch.allclose(out_custom, out_ref, atol=1e-5), "Fallback conv2d with dilation failed."

# Issue 3: Kernel only supports float32.
# Passing double precision tensors should lead to incorrect memory accesses (or a runtime error).
def test_dtype_handling():
    module = build_kernel()
    x = torch.randn(1, 1, 10, 10, device="cuda", dtype=torch.float64)
    weight = torch.randn(1, 1, 3, 3, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # Likely failure because the kernel assumes float32 and uses __ldg on float pointers.
        module.forward(x, weight, None, 1, 1, 1, 1)

# Issue 4: Race condition with constant memory.
# Launch multiple concurrent calls to forward, each copying a weight tensor into constant memory.
def test_race_condition():
    module = build_kernel()
    # Use a small weight tensor that fits in constant memory.
    x = torch.randn(8, 1, 32, 32, device="cuda", dtype=torch.float32)
    weight = torch.randn(1, 1, 3, 3, device="cuda", dtype=torch.float32)
    outputs = []

    def run_forward():
        out = module.forward(x, weight, None, 1, 1, 1, 1)
        outputs.append(out.cpu())

    threads = []
    for _ in range(10):
        t = threading.Thread(target=run_forward)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # Compute a reference result using PyTorch's conv2d.
    ref = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=1)
    for out in outputs:
        assert torch.allclose(out, ref.cpu(), atol=1e-5), "Concurrent execution yielded mismatched outputs."

# Issue 5: Lack of error checking for cudaMemcpyToSymbol and kernel launch errors.
# For example, if the kernel parameters result in out-of-bound accesses, the kernel may fail without proper error-checking.
def test_error_checking():
    module = build_kernel()
    # Intentionally force an error by providing a weight kernel that is too large relative
    # to the input dimensions so that the computed output dimensions are invalid.
    x = torch.randn(1, 1, 5, 5, device="cuda", dtype=torch.float32)
    weight = torch.randn(1, 1, 7, 7, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Expect a runtime error due to illegal indexing or kernel launch failure.
        module.forward(x, weight, None, 1, 0, 1, 1)
