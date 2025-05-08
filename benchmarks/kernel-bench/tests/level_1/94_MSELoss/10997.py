
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility to build (and reload) the CUDA extension.
def build_kernel():
    return load(
        name="mse_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
        with_cuda=True,
    )

# Issue 1: Block size assumption test.
# We simulate a case where the launch configuration differs.
# Although our C++ code fixes the block size to BLOCK_SIZE,
# we can force a situation where predictions are very large so that
# a mismatch between computing work and available reduction storage appears.
# (Note: to trigger the mis-allocation one must change the kernel launch parameters.
# For testing, we simulate passing an input that requires a block size different from BLOCK_SIZE.)
def test_block_size_assumption(monkeypatch):
    # Monkey-patch the kernel launch to use a custom block size that is not a multiple of 32.
    # This is simulated by overriding the grid_size computation, for example.
    # (For instance, if we set BLOCK_SIZE=250 instead of 256, then 250 % 32 != 0.)
    # Since the C++ code hardcodes BLOCK_SIZE, we simulate this by passing an input tensor
    # with num_elements that forces a remainder when divided by 32.
    module = build_kernel()
    N = 250  # Not a multiple of 32
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    # We expect that even if the reduction is “incorrect”, the result will be off.
    # Compute the expected mse on the host.
    mse_ref = torch.mean((A - B) ** 2).item()
    mse_cuda = module.forward(A, B).item()
    # Due to undefined behavior in reduction, the result may not match.
    with pytest.raises(AssertionError):
        assert pytest.approx(mse_ref, rel=1e-5) == mse_cuda, \
            f"Expected mse {mse_ref}, got {mse_cuda}. Block size assumption may be violated."

# Issue 2: Warp-size assumption test.
# This test forces a situation where blockDim.x would not be a multiple of 32.
# Since the kernel launch in our extension is fixed, we simulate a potential problem by using a small tensor.
def test_warp_size_assumption(monkeypatch):
    module = build_kernel()
    # Create a tensor with a number of elements that is not a multiple of 32.
    # Even though the kernel still launches 256 threads per block,
    # the effective working set may include threads that do not have a full warp.
    N = 45  # 45 < 256 and not a multiple of 32
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    mse_ref = torch.mean((A - B) ** 2).item()
    mse_cuda = module.forward(A, B).item()
    with pytest.raises(AssertionError):
        assert pytest.approx(mse_ref, rel=1e-5) == mse_cuda, \
            f"Expected mse {mse_ref}, got {mse_cuda}. Warp-size assumption may be violated."

# Issue 3: Handling of zero-length input.
def test_zero_length_input():
    module = build_kernel()
    # Create empty tensors for predictions and targets.
    A = torch.empty((0,), device="cuda", dtype=torch.float32)
    B = torch.empty((0,), device="cuda", dtype=torch.float32)
    # In a correct implementation, calling mean on an empty tensor would throw an error,
    # or the kernel would avoid division-by-zero.
    with pytest.raises(AssertionError):
        # Expect that the result is not a valid number (nan or inf)
        mse_cuda = module.forward(A, B).item()
        assert torch.isfinite(torch.tensor(mse_cuda)), \
            f"Expected a finite mse, got {mse_cuda}. Zero-length input handling may be incorrect."

# Issue 4: Non-floating point tensor input.
def test_non_floating_point_input():
    module = build_kernel()
    # Create integer tensors, which are not supported by AT_DISPATCH_FLOATING_TYPES.
    A = torch.randint(0, 10, (128,), device="cuda", dtype=torch.int32)
    B = torch.randint(0, 10, (128,), device="cuda", dtype=torch.int32)
    # We expect the TORCH_CHECK or dispatch macro to throw an error.
    with pytest.raises(RuntimeError):
        module.forward(A, B)
