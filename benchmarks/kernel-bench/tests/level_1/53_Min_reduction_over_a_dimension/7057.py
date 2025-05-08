
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel(extra_cuda_cflags=None):
    if extra_cuda_cflags is None:
        extra_cuda_cflags = ["-O3", "--use_fast_math"]
    cuda_module = load(
        name="test_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_half_precision():
    # Issue 1: Using half precision.
    # We use half-precision input. The kernel uses std::numeric_limits
    # which is not appropriate for torch.float16 and should trigger a failure.
    kernel_module = build_kernel()
    x = torch.randn(16, 32, 32, device='cuda', dtype=torch.float16)
    # Expect the kernel to raise an error or produce an incorrect result.
    with pytest.raises(RuntimeError):
        out = kernel_module.forward(x, 1)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_empty_reduction_dimension():
    # Issue 2: Empty reduction dimension.
    # Create an input with an empty reduction dimension.
    kernel_module = build_kernel()
    # Let dimension 1 be empty.
    x = torch.randn(16, 0, 32, device='cuda', dtype=torch.float32)
    # The behavior is undefined; we expect the kernel either to error or return a tensor full of std::numeric_limits max.
    # Here we check if the output is filled with the max value.
    out = kernel_module.forward(x, 1)
    torch.cuda.synchronize()
    # Compute the max float value for float32.
    max_val = torch.tensor(torch.finfo(torch.float32).max, device='cuda')
    if out.numel() > 0:
        assert torch.all(out == max_val), f"Expected all elements to be {max_val} for empty reduction, got {out}"
    else:
        pytest.skip("Output is empty, test inconclusive for empty reduction dimension.")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_incorrect_shuffle_mask_with_small_block():
    # Issue 3: Block size smaller than warp size leads to incorrect warp-level reduction
    # because the kernel uses a fixed mask 0xffffffff.
    kernel_module = build_kernel()
    # Create an input tensor that will be reduced along dim=1.
    # Use a block_size less than 32 to force an incomplete warp.
    x = torch.randn(16, 64, 32, device='cuda', dtype=torch.float32)
    # Compute the expected result using PyTorch's built-in reduction.
    expected = torch.min(x, dim=1)[0]
    # Launch the kernel with an intentionally small block_size.
    out = kernel_module.forward(x, 1, block_size=16)
    torch.cuda.synchronize()
    # We expect the result to be incorrect due to the shuffle mask issue.
    if torch.allclose(expected, out, atol=1e-5):
        pytest.fail("Kernel reduction produced correct result with block_size < warpSize, but expected an error due to masked shuffle issue.")
    else:
        # The output is wrong as expected.
        assert not torch.allclose(expected, out, atol=1e-5), "Kernel output unexpectedly matches torch.min despite improper block configuration."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_no_cuda_error_checking():
    # Issue 4: The kernel does not check for CUDA errors after launch.
    # We can simulate this by providing an input tensor on CPU.
    kernel_module = build_kernel()
    x = torch.randn(16, 32, 32, device='cpu', dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input must be a CUDA tensor"):
        _ = kernel_module.forward(x, 1)
        torch.cuda.synchronize()
