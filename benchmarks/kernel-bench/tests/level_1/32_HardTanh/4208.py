
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the CUDA extension from kernel.cu.
    cuda_module = load(
        name="hardtanh_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_module():
    return build_kernel()

def test_non_float32_dtype(cuda_module):
    # Issue: Kernel only supports float32 but not other floating types.
    min_val, max_val = -1.0, 1.0
    # Create a tensor in float64.
    x = torch.randn(1024, device='cuda', dtype=torch.float64)
    # Expected result computed with PyTorch (using its proper hardtanh/clamp implementation).
    expected = torch.clamp(x, min=min_val, max=max_val)
    # Pass the double tensor to our kernel implementation.
    # Since the kernel always treats the input as float32,
    # the resulting output will be misinterpreted and wrong.
    out = cuda_module.forward(x, min_val, max_val)
    torch.cuda.synchronize()
    # Because the kernel code forcefully uses float pointers,
    # its output dtype will be float32.
    assert out.dtype == torch.float32, "Kernel output dtype should be float32 due to the hard-coded cast."
    # Convert the expected result to float32 for comparison.
    expected = expected.to(torch.float32)
    # The output is expected to be incorrect when the input is not float32.
    with pytest.raises(AssertionError):
        assert torch.allclose(out, expected, atol=1e-5), (
            "Kernel incorrectly processed a double tensor as float32; "
            "the output should differ from the expected result."
        )

def test_non_contiguous_input(cuda_module):
    # Issue: Kernel assumes contiguous input.
    min_val, max_val = -1.0, 1.0
    # Create a contiguous tensor.
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    # Make a non-contiguous tensor by transposing.
    x_nc = x.t()
    # Expected result using PyTorch’s clamp (which works regardless of contiguity).
    expected = torch.clamp(x_nc, min=min_val, max=max_val)
    # Run the CUDA kernel on the non-contiguous tensor.
    out = cuda_module.forward(x_nc, min_val, max_val)
    torch.cuda.synchronize()
    # The kernel does not adjust for non-contiguous memory so the output is expected to be wrong.
    with pytest.raises(AssertionError):
        assert torch.allclose(out, expected, atol=1e-5), (
            "Kernel incorrectly processed a non-contiguous tensor; "
            "the output should differ from the expected result."
        )
