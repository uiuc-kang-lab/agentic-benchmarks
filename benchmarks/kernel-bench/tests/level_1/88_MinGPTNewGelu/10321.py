
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="gelu_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def gelu_reference(x):
    # Reference GELU implementation as used in the PyTorch code.
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * torch.pow(x, 3))))

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

def test_dtype_mismatch(kernel_module):
    # Issue 1: Passing a double (float64) tensor should either be unsupported or yield wrong results.
    x = torch.randn(1024, device='cuda', dtype=torch.double)
    # The host function does not check the tensor type, so it will reinterpret the data pointer.
    # This will result in incorrect outputs. We test for a significant deviation.
    y = kernel_module.forward(x)
    # Compute a reference result by converting to float and then back to double.
    x_float = x.float()
    y_ref = gelu_reference(x_float).double()
    # Expect the outputs NOT to be nearly equal.
    assert not torch.allclose(y, y_ref, atol=1e-3), \
        "Kernel incorrectly processed a double input. It should either reject or handle non-float32 types properly."

def test_non_contiguous(kernel_module):
    # Issue 2: Non-contiguous inputs should be checked and rejected.
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    # Make x non-contiguous (e.g., by transposing)
    x_non_contig = x.t()
    with pytest.raises(RuntimeError):
        # The host function explicitly checks for contiguity.
        _ = kernel_module.forward(x_non_contig)

def test_fixed_tiling(kernel_module):
    # Issue 3: The kernel uses a fixed unroll factor and tiling strategy which may not optimally handle
    # input sizes that are not multiples of the tile size.
    # Use an input size that is not divisible by (blockDim.x * unroll factor); default threads = 256, unroll = 4 (tile size = 1024).
    # Here we choose a tensor with a total number of elements that is not a multiple of 1024.
    total_elements = 12345  # Not a multiple of 1024
    x = torch.randn(total_elements, device="cuda", dtype=torch.float32)
    y = kernel_module.forward(x)
    y_ref = gelu_reference(x)
    # Although the kernel bounds-checks each work-item, the fixed tiling may lead to suboptimal computation;
    # here we simply check that the computed result is correct.
    assert torch.allclose(y, y_ref, atol=1e-5), \
        "Kernel output is incorrect for an input size not divisible by the tile size. The tiling/unrolling logic may need to be more general."

