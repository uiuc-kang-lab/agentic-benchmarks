
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to compile and load the CUDA kernel extension
def build_kernel():
    cuda_module = load(
        name="frobenius_norm_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Division by zero when input tensor has zero Frobenius norm.
def test_zero_norm_division():
    my_kernel = build_kernel()
    # Create an input tensor that is all zeros
    x = torch.zeros(1024, device="cuda", dtype=torch.float32)
    # When dividing by zero, the expected output will be NaN or Inf.
    out = my_kernel.forward(x)
    # Check if any element in the output is NaN or infinite.
    assert torch.isnan(out).any() or torch.isinf(out).any(), (
        "Expected NaN or Inf in output when input tensor has zero norm"
    )

# Issue 2: Input tensor type check and contiguity check.
def test_input_tensor_dtype():
    my_kernel = build_kernel()
    # Create a tensor with a different floating type
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="Input must be float32"):
        my_kernel.forward(x)

def test_input_tensor_contiguity():
    my_kernel = build_kernel()
    # Create a contiguous tensor then make it non-contiguous via a transpose or slicing.
    x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    non_contig = x.t()  # transpose makes it non-contiguous
    with pytest.raises(RuntimeError, match="contiguous"):
        my_kernel.forward(non_contig)

# Issue 3: Fixed shared memory allocation may not work for arbitrary block sizes.
# Although the module always launches kernels with threads = 256,
# we can simulate a stress test where many warps per block are desirable by fabricating a large input.
# (In a more general kernel, one would allow the block size to be chosen. This test
#  is provided to alert developers that the kernel may not generalize if blockDim.x > 256.)
def test_large_input_multiple_warps():
    my_kernel = build_kernel()
    # Create an input tensor with many elements so that—even with loop iterations—
    # each thread will process many elements.
    # Note: The kernel launch in forward() always uses 256 threads per block,
    # so this test will pass as long as the per-thread loop works correctly.
    size = 256 * 70  # 70 blocks worth of threads if processed in one iteration,
                     # but then each thread loops to cover the extra elements.
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    out = my_kernel.forward(x)
    # Compare the result to a PyTorch reference computation.
    norm = torch.norm(x, p='fro')
    expected = x / norm
    # This test is expected to fail if shared memory reduction is not properly generalized.
    assert torch.allclose(out, expected, atol=1e-5), "Output does not match expected normalization."

