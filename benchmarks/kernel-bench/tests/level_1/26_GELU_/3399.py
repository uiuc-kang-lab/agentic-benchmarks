
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="gelu_cuda_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_double_precision_issue():
    # Issue 1: Using double precision should be processed using erf and not erff.
    # Here we create a double precision tensor. The custom kernel uses erff which will produce
    # an incorrect result when computed in double precision as compared to torch.nn.functional.gelu.
    kernel = build_kernel()
    x = torch.randn(128, 256, dtype=torch.double, device='cuda')
    out_kernel = kernel.forward(x)
    out_ref = torch.nn.functional.gelu(x)
    # We expect a significant difference due to the wrong intrinsic call.
    max_diff = (out_kernel - out_ref).abs().max().item()
    assert max_diff > 1e-3, f"Double precision issue not detected: max diff {max_diff}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_alignment_issue():
    # Issue 2: The vectorized kernel uses float4 loads/stores that assume 16-byte alignment.
    # We create a float32 tensor and then slice it to break the alignment.
    base = torch.randn(128, 257, dtype=torch.float32, device='cuda')
    # Slicing to force a non-16-byte-aligned memory pointer.
    x = base[:, 1:]
    # Ensure x is contiguous. The slicing offsets the underlying data pointer.
    if not x.is_contiguous():
        x = x.contiguous()
    kernel = build_kernel()
    out_kernel = kernel.forward(x)
    out_ref = torch.nn.functional.gelu(x)
    max_diff = (out_kernel - out_ref).abs().max().item()
    assert max_diff > 1e-4, f"Alignment issue not triggered: max diff {max_diff}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_missing_kernel_error_checking():
    # Issue 3: The kernel launches do not check for errors. We can simulate a scenario
    # that might produce an error by passing an empty tensor (numel=0) or malformed shape.
    # An empty tensor should run without processing but might trigger a kernel launch error
    # if not handled properly.
    kernel = build_kernel()
    x = torch.empty(0, dtype=torch.float32, device='cuda')
    try:
        out = kernel.forward(x)
        # Synchronize to force any asynchronous errors to surface.
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel launch error was not checked and surfaced as an exception: {e}")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    # Issue 4: The kernel does not enforce contiguous input.
    # Create a non-contiguous tensor by transposing a contiguous tensor.
    x_contig = torch.randn(64, 128, dtype=torch.float32, device='cuda')
    x = x_contig.t()  # now non-contiguous
    kernel = build_kernel()
    out_kernel = kernel.forward(x)
    out_ref = torch.nn.functional.gelu(x)
    max_diff = (out_kernel - out_ref).abs().max().item()
    # Expect the non-contiguous tensor to cause a problem, so the difference should be large.
    assert max_diff > 1e-4, f"Non-contiguous issue not triggered: max diff {max_diff}"
