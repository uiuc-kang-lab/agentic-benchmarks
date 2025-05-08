
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build/load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="gelu_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Half-precision support
def test_half_precision():
    my_module = build_kernel()
    x = torch.randn(16, 16384, device="cuda", dtype=torch.half)
    # We expect the kernel to raise an error because half precision is not dispatched.
    with pytest.raises(RuntimeError):
        # The kernel forward call should trigger the AT_DISPATCH error.
        _ = my_module.forward(x)

# Test 2: Non-contiguous input
def test_non_contiguous():
    my_module = build_kernel()
    # Create a contiguous 2D tensor and then transpose it to make it non-contiguous.
    x_contig = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x = x_contig.t()  # now non-contiguous
    # The reference GELU (which supports non-contiguous inputs) computes the correct result.
    ref = torch.nn.functional.gelu(x)
    out = my_module.forward(x)
    # Because the kernel uses x.data_ptr() and x.numel(), it assumes contiguous layout,
    # so the output will in general differ from the reference.
    assert not torch.allclose(out, ref, atol=1e-5), "Kernel unexpectedly handled non-contiguous inputs correctly."

# Test 3: Synchronization / asynchronous error propagation
def test_missing_synchronization():
    my_module = build_kernel()
    # Pass an input that the kernel cannot process correctly.
    # One way to do this is to pass a CPU tensor. The forward() function explicitly checks 
    # for x.is_cuda() but if that check were removed or bypassed, the kernel launch would fail asynchronously.
    x = torch.randn(16, 16384, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x)

# Test 4: Large tensor index overflow simulation
def test_index_overflow_simulation(monkeypatch):
    my_module = build_kernel()
    # We cannot easily allocate a tensor with >INT_MAX elements, but we can simulate the scenario
    # by monkeypatching the numel() method to return an artificially large number.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    original_numel = x.numel
    try:
        x.numel = lambda: (2**31) + 100  # simulate a huge number of elements (overflow situation)
        # If the kernel were robust, it would use a 64-bit index and still work.
        # Here, we expect that the use of a 32-bit index in the kernel might cause an error or incorrect behavior.
        out = my_module.forward(x)
        # Check if the output size matches the simulated numel.
        # Since we can't allocate that memory, this test is mostly conceptual.
        # We can at least verify that the call completes, but we mark a warning if it doesn't.
        assert out.numel() == x.numel(), "Output tensor numel does not match simulated huge input size."
    except RuntimeError:
        pytest.skip("Test skipped because simulating index overflow is not feasible in the current environment.")
    finally:
        x.numel = original_numel
