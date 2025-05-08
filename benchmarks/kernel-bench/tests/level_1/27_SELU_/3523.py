
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="selu_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Non-contiguous tensors.
# The CUDA kernel flattens the tensor and ignores its strides, meaning that when a non-contiguous tensor is passed,
# the computed output will not match torch.selu()’s result.
def test_non_contiguous_input():
    module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous (e.g. by transposing a 2D tensor)
    x = torch.randn(128, 256, device="cuda")
    x_non_contig = x.t()  # This transpose makes it non-contiguous
    # Expected output using torch.selu which handles strides properly
    expected = torch.selu(x_non_contig)
    # Output from our custom CUDA kernel; note: our kernel flattens the tensor ignoring strides.
    out = module.forward(x_non_contig)
    torch.cuda.synchronize()
    # The results will differ because of the wrong memory access due to non-contiguity.
    with pytest.raises(AssertionError):
        # We assert that result should match the expected SELU, but it will not.
        assert torch.allclose(out, expected, atol=1e-5), "Kernel incorrectly processed a non-contiguous tensor."

# Issue 2: Lack of half-precision (float16) support.
# A half-precision input should trigger an error because AT_DISPATCH_FLOATING_TYPES does not cover float16.
def test_half_precision_input():
    module = build_kernel()
    x = torch.randn(1024, device="cuda").half()
    with pytest.raises(RuntimeError):
        # Expect a RuntimeError during dispatch due to unsupported data type (float16).
        module.forward(x)
        torch.cuda.synchronize()

# Issue 3: Lack of error checking after kernel launch.
# While the kernel pre-checks (via TORCH_CHECK) that the input is a CUDA tensor, passing a CPU tensor should trigger an error.
def test_input_not_on_cuda():
    module = build_kernel()
    x = torch.randn(1024, device="cpu")
    with pytest.raises(RuntimeError):
        # Expect a RuntimeError because TORCH_CHECK(input.is_cuda()) should fail.
        module.forward(x)
