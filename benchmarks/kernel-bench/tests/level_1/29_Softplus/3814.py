
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the extension module from kernel.cu
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# 1. Test for inconsistent warp-synchronous behavior.
#    Use an input size that is not a multiple of the block dimension
#    so that the last warp is partially active.
def test_warp_divergence():
    my_module = build_kernel()
    # Create a tensor with size not divisible by block size 256.
    N = 1023  # 1023 is not a multiple of 256, ensuring a partial warp in the last block.
    # Use values distributed across the softplus decision regions.
    # Use a mix of high, low, and intermediate values.
    x = torch.empty(N, device="cuda", dtype=torch.float32)
    # First third: low values < -20, second third: intermediate, third: high values > 20.
    x[:N//3] = -30.0
    x[N//3:2*N//3] = 0.0
    x[2*N//3:] = 30.0

    # Run our CUDA kernel and the reference torch implementation.
    y_cuda = my_module.forward(x)
    y_ref = torch.nn.functional.softplus(x)

    torch.cuda.synchronize()
    # Allow a slight tolerance due to branch approximations.
    assert torch.allclose(y_cuda, y_ref, atol=1e-5), \
        f"Warp divergence test did not match: max error {(y_cuda - y_ref).abs().max().item()}"

# 2. Test for hard-coded threshold precision.
#    Provide a double precision input tensor to see if the thresholds are applied correctly.
def test_double_threshold():
    my_module = build_kernel()
    # Create a double precision tensor with borderline values around the threshold.
    # Because the kernel uses 20.0f constants, the decisions for values near 20 might be off.
    x = torch.tensor([-21.0, -20.0, 0.0, 20.0, 21.0], device="cuda", dtype=torch.float64)
    y_cuda = my_module.forward(x)
    y_ref = torch.nn.functional.softplus(x)

    torch.cuda.synchronize()
    # This test may fail if the threshold comparisons are incorrect for double.
    assert torch.allclose(y_cuda, y_ref, atol=1e-8), \
        f"Double threshold test failed: CUDA output {y_cuda} vs reference {y_ref}"

# 3. Test for non-contiguous input memory.
#    Pass a non-contiguous tensor into the kernel. The kernel assumes contiguous memory,
#    so the result may be incorrect.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous input tensor then get a non-contiguous view using transpose.
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # Transpose makes it non-contiguous.

    # Expect that our kernel (which assumes contiguity) will not compute the correct result.
    y_cuda = my_module.forward(x_noncontig)
    y_ref = torch.nn.functional.softplus(x_noncontig)

    torch.cuda.synchronize()
    # We assert that the outputs are not equal to flag the issue.
    assert not torch.allclose(y_cuda, y_ref, atol=1e-5), \
        "Non-contiguous input test: Kernel unexpectedly produced correct results on non-contiguous data."

# 4. Test for kernel error checking (lack thereof).
#    We deliberately supply an invalid tensor (e.g. wrong device) to trigger a runtime error.
def test_invalid_tensor_device():
    my_module = build_kernel()
    # Create a tensor on CPU instead of CUDA.
    x_cpu = torch.randn(1024, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Expect the kernel launch to error out because the input is not on CUDA.
        my_module.forward(x_cpu)
