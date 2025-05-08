
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="kernel_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

kernel_module = build_kernel()

def run_forward(x, dim):
    # Wrap the CUDA kernel invocation.
    return kernel_module.forward(x, dim)

def test_dtype_check():
    # Issue 1: The kernel does not validate the input tensor's dtype.
    # Create a double tensor, but the kernel assumes float, so this should trigger an error.
    x = torch.randn(16, 32, 32, dtype=torch.double, device='cuda')
    with pytest.raises(RuntimeError, match=".*"):
        out = run_forward(x, 1)
        torch.cuda.synchronize()

def test_contiguous_check():
    # Although not one of the listed issues for this kernel,
    # we can at least verify that the CHECK_CONTIGUOUS macro fires.
    # Create a non-contiguous tensor by transposing.
    x = torch.randn(16, 32, 32, device='cuda', dtype=torch.float32).transpose(0, 1)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        out = run_forward(x, 1)
        torch.cuda.synchronize()

def test_incorrect_indexing():
    # Issue 2: The kernel's indexing logic is too simplistic.
    # Here we use a tensor with more than 2 dimensions and pick a reduction dimension
    # that is not the first dimension. The expected result from torch.prod will differ.
    x = torch.randn(4, 5, 6, device='cuda', dtype=torch.float32)
    expected = torch.prod(x, dim=1)
    out = run_forward(x, 1)
    torch.cuda.synchronize()
    # Because the kernel miscomputes the base pointer for the reduction slice,
    # its output should not match the expected result.
    assert not torch.allclose(out, expected, atol=1e-5), (
        "Kernel unexpectedly produced correct result on a non-standard reduction dimension."
    )

def test_empty_reduction():
    # Issue 3: The kernel does not handle empty reduction dimensions.
    # For an empty reduction dimension, torch.prod returns 1.
    # Create a tensor where the reduction dimension size is zero.
    x = torch.randn(10, 0, 5, device='cuda', dtype=torch.float32)
    expected = torch.ones(x.shape[0], x.shape[2], device='cuda', dtype=torch.float32)
    out = run_forward(x, 1)
    torch.cuda.synchronize()
    # If the kernel handled empty dimensions correctly, out should equal expected.
    if torch.allclose(out, expected, atol=1e-5):
        pytest.fail("Kernel did not exhibit an error on an empty reduction dimension when it should have.")

