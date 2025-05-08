
import pytest
import torch
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="softsign_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# The expected CPU implementation of softsign activation.
def softsign_cpu(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.abs(x))

# Issue 1: Unsupported data types.
# The kernel assumes float32 but does not check. Passing a float64 tensor will cause the memory to be reinterpreted.
def test_non_float_input():
    device = "cuda"
    # Create a double tensor
    x = torch.randn(1024, device=device, dtype=torch.float64)
    cpu_out = softsign_cpu(x)
    mod = build_kernel()
    # The kernel launch does not check type so it will treat the double array as float.
    # We expect the resulting values to be numerically far off from the expected ones.
    out = mod.forward(x)
    torch.cuda.synchronize()
    # Because of reinterpretation, the output will not match the correct softsign.
    with pytest.raises(AssertionError):
        # We force a failure if mistakenly the output would match the expected output.
        assert torch.allclose(out.double(), cpu_out, atol=1e-5)

# Issue 2: Unused aligned_threads (i.e. lack of proper thread block alignment).
# Although this may not change correctness, we simulate it by verifying that when the number of elements is not a multiple
# of the expected warp size, the output is computed correctly (hinting that the intended alignment is not enforced).
def test_unaligned_input_size():
    device = "cuda"
    # Choose a size that is not a multiple of 32 (a warp size)
    N = 1001  
    x = torch.randn(N, device=device, dtype=torch.float32)
    cpu_out = softsign_cpu(x)
    mod = build_kernel()
    out = mod.forward(x)
    torch.cuda.synchronize()
    # Even though aligned_threads was computed but not used,
    # the kernel still computes the correct result.
    assert torch.allclose(out, cpu_out, atol=1e-5), "Output mismatch for input size not a multiple of warp size."

# Issue 3: Lack of kernel error checking.
# We attempt to trigger a situation that may produce a kernel error.
# One way is to request a grid configuration that exceeds device limits.
# (Note: this test may depend on the GPU, so we choose an input so large that the launch may fail.)
def test_kernel_launch_error():
    device = "cuda"
    mod = build_kernel()
    # Creating an exceedingly large tensor to potentially trigger a kernel launch configuration error.
    try:
        # Note: This tensor is very large and may not be allocated on many GPUs.
        # Adjust size if necessary to reliably trigger an error on your system.
        huge_num = 2**28  # Adjust this to force grid dimension issues on some devices.
        x = torch.randn(huge_num, device=device, dtype=torch.float32)
    except RuntimeError:
        pytest.skip("Skipping kernel launch error test since huge allocation is not possible on this device.")
    # The lack of error checking means kernel errors may only surface upon synchronization.
    with pytest.raises(RuntimeError):
        out = mod.forward(x)
        # Force synchronization to capture any kernel launch errors.
        torch.cuda.synchronize()

# Issue 4: Rigid block configuration.
# We test the kernel on an input whose size forces the fixed block size to produce significant workload imbalance.
def test_fixed_block_config():
    device = "cuda"
    mod = build_kernel()
    # Select a size that is not ideally matching the fixed block size 1024,
    # e.g. number of elements is moderately larger than 1024 so many threads will process zero or one element.
    N = 5000  
    x = torch.randn(N, device=device, dtype=torch.float32)
    cpu_out = softsign_cpu(x)
    out = mod.forward(x)
    torch.cuda.synchronize()
    assert torch.allclose(out, cpu_out, atol=1e-5), "Output mismatch due to fixed block configuration."

