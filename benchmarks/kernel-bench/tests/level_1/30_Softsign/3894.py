
import os
import torch
import pytest
from torch.utils.cpp_extension import load

# Build and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Kernel does not verify input type.
# Test: Pass a torch.float64 tensor. The kernel always reads as float.
# Expected behavior: The output will be computed as if the underlying bits were float32,
# leading to data corruption. In our test we compare with the expected result computed in float64.
def test_input_wrong_dtype():
    module = build_kernel()
    # create a tensor of type float64
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    out = module.forward(x)
    # expected softsign computed in float64 (correct handling)
    expected = x / (1 + torch.abs(x))
    # Because the kernel misinterprets the data (reads float32 elements), we expect a mismatch.
    # We check that the outputs are not close.
    with pytest.raises(AssertionError):
        # This assertion should fail because the kernel output will be incorrect.
        assert torch.allclose(out.to(torch.float64), expected, atol=1e-5)

# Issue 2: The kernel lacks error checking after launch.
# Test: Force a kernel launch error by monkey-patching torch.cuda.synchronize.
def test_kernel_launch_error_checking(monkeypatch):
    module = build_kernel()
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    # Define a fake torch.cuda.synchronize that always raises an error.
    def fake_synchronize():
        raise RuntimeError("Fake kernel launch error")
    monkeypatch.setattr(torch.cuda, "synchronize", fake_synchronize)
    # Calling the forward should eventually lead to an error during synchronization,
    # yet the kernel function itself does not catch errors.
    with pytest.raises(RuntimeError, match="Fake kernel launch error"):
        out = module.forward(x)
        # Force a synchronize to propagate the error (simulated).
        torch.cuda.synchronize()

# Issue 3: The __constant__ variable BLOCK_SIZE is defined but never used.
# Test: Check that the source file 'kernel.cu' defines BLOCK_SIZE but does not actually use it in the kernel launch.
def test_unused_constant():
    # Read the kernel source code
    source_file = os.path.join(os.path.dirname(__file__), "kernel.cu")
    with open(source_file, "r") as f:
        content = f.read()
    # Count occurrences of BLOCK_SIZE outside of its definition
    # We expect to see its definition but not its use for grid/block configuration.
    definition_occurrences = content.count("__constant__ int BLOCK_SIZE")
    usage_occurrences = content.count("BLOCK_SIZE") - definition_occurrences
    # The test detects that while BLOCK_SIZE is defined, it is never used.
    assert definition_occurrences >= 1, "BLOCK_SIZE should be defined in the file."
    assert usage_occurrences == 0, "BLOCK_SIZE is defined but never used."
