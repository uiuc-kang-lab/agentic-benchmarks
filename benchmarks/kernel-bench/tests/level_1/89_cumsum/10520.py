
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Ensure we compile the kernel from kernel.cu
    cuda_module = load(
        name="cumsum_cuda",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test Case 1: Trigger excessive thread block size issue.
def test_large_inner_size():
    # We want to force the kernel launch to use a thread block size greater than the max allowed.
    # Typically the maximum threads per block is 1024.
    # We design a tensor such that inner_size = number of threads > 1024.
    # Note: The forward function in the kernel computes:
    #   outer_size = product(x.size(0) .. x.size(dim-1))
    #   inner_size = product(x.size(dim+1) ... x.size(ndim-1))
    #   stride = x.size(dim)
    # To force inner_size large, we set dim=0 with tensor shape (M, N)
    # where inner_size = N. Let N = 2048 (>1024) and M = 1.
    cuda_module = build_kernel()
    # Create a tensor of shape (1, 2048) which is contiguous and of type float32.
    x = torch.randn(1, 2048, dtype=torch.float32, device="cuda")
    dim = 0  # cumulative sum along dim 0, so blockDim becomes inner_size = 2048.
    with pytest.raises(RuntimeError):
        # Expect that launching with an invalid block size throws a CUDA launch error.
        output = cuda_module.forward(x, dim)
        torch.cuda.synchronize()

# Test Case 2: Trigger unsupported dtype (non-float32) issue.
def test_unsupported_dtype():
    cuda_module = build_kernel()
    # Create a tensor of type double which is not supported by the kernel.
    # The kernel expects float32 so passing double should raise an error.
    x = torch.randn(10, 10, dtype=torch.double, device="cuda")
    dim = 1
    with pytest.raises(Exception):
        # Depending on how PyTorch extension invokes the kernel,
        # this may raise an error about incompatible pointer type.
        output = cuda_module.forward(x, dim)
        torch.cuda.synchronize()

# Test Case 3: Trigger potential misalignment issue.
def test_unaligned_memory():
    cuda_module = build_kernel()
    # Although PyTorch tensors are usually aligned, we can force an unaligned tensor via slicing.
    # By taking a slice from a larger tensor, we can potentially break the pointer alignment.
    base = torch.randn(1024 + 1, dtype=torch.float32, device="cuda")
    # Create a slice that is likely misaligned (skip the first element).
    x_unaligned = base[1:].clone().view(1, -1)
    dim = 1
    # The kernel might work correctly if alignment is not critical,
    # but in a more general context this test should help reveal any potential issues.
    # Here, we check that the output is different from the expected cumsum if misalignment causes issues.
    # For testing purposes we compare with PyTorch's own cumsum.
    output = cuda_module.forward(x_unaligned, dim)
    torch.cuda.synchronize()
    expected = torch.cumsum(x_unaligned, dim=dim)
    # In a faulty scenario the output may differ significantly.
    assert not torch.allclose(output, expected, atol=1e-5), "Kernel unexpectedly handled unaligned memory correctly."

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
