
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_kernel():
    return build_kernel()

def test_non_contiguous_input(cuda_kernel):
    # Issue 1: non‐contiguous tensor support
    # Create a tensor and then make it non-contiguous by transposing dimensions.
    original = torch.randn(8, 16, 32, device="cuda")
    non_contig = original.transpose(0, 1)  # now non contiguous
    # Compute reference using torch built-in max reduction
    ref = torch.max(non_contig, dim=0)[0]
    # The kernel expects the reduction dimension as an integer; here we reduce dimension 0.
    output = cuda_kernel.forward(non_contig, 0)
    # Because the kernel assumes contiguity, its result will likely differ.
    # We intentionally check that the error is significant.
    diff = (output - ref).abs().max().item()
    assert diff > 1e-3, "Kernel incorrectly handled non-contiguous input!"

def test_non_power_of_two_block(cuda_kernel):
    # Issue 2: non-power-of-two block size assumption.
    # Create a tensor where the reduction dimension size is not a power of two.
    # For example, use dim_size = 150 (not power-of-two), with a layout that is contiguous.
    batch, h, w = 4, 150, 32
    x = torch.randn(batch, h, w, device="cuda")
    # Reduce along dimension 1
    ref = torch.max(x, dim=1)[0]
    output = cuda_kernel.forward(x, 1)
    # With the mis-assumed reduction algorithm, the result may be wrong.
    # We check that the maximum difference is non-negligible.
    diff = (output - ref).abs().max().item()
    assert diff > 1e-3, "Kernel appears to work correctly for non-power-of-two block sizes, but it was expected to fail!"

def test_empty_reduction_dim(cuda_kernel):
    # Issue 3: empty reduction dimension (dim_size == 0).
    # Create a tensor with an empty reduction dimension.
    x = torch.randn(4, 0, 10, device="cuda")
    # The reference torch.max will error; we expect our kernel to not correctly handle empty dims.
    with pytest.raises(RuntimeError):
        _ = cuda_kernel.forward(x, 1)

def test_integer_dtype(cuda_kernel):
    # Issue 4: Only floating point and half are dispatched.
    # Create a tensor with integer type.
    x = torch.randint(0, 100, (8, 16, 32), device="cuda", dtype=torch.int32)
    # Compute reference using torch.max reduction
    ref = torch.max(x, dim=1)[0]
    # The kernel (via AT_DISPATCH_FLOATING_TYPES_AND_HALF) does not support int32, so either it errors or returns wrong result.
    try:
        output = cuda_kernel.forward(x, 1)
    except Exception as e:
        # If an error occurs then this confirms the lack of support.
        return
    # If no error, then compare results; they should differ because the kernel wasn’t meant for int types.
    diff = (output.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    assert diff > 1e-3, "Kernel incorrectly processed integer dtype input!"
