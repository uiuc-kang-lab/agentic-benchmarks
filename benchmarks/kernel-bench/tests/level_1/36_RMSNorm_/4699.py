
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the CUDA extension module from kernel.cu.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Non-contiguous tensor
def test_non_contiguous_tensor():
    # Create a contiguous input then make it non-contiguous.
    x = torch.randn(16, 64, 256, 256, device="cuda")
    x_noncontig = x.transpose(1, 2)  # now non-contiguous and feature dimension no longer is dim=1
    mod = build_kernel()
    # Depending on the error-checking strategy in C++ code, the kernel may produce wrong results.
    # We check if the output is different than the expected (using the original python normalization)
    out_cuda = mod.forward(x_noncontig, 1e-5)
    # Use PyTorch’s native normalization (normalize along what index? here we assume fixed axis 1)
    # But note that in x_noncontig the assumed normalization axis is lost.
    out_ref = x_noncontig / torch.sqrt(torch.mean(x_noncontig**2, dim=1, keepdim=True) + 1e-5)
    assert not torch.allclose(out_cuda, out_ref, atol=1e-3), \
        "Kernel did not fail for non-contiguous tensor as expected."

# Issue 2: Input tensor of too low rank
def test_low_rank_input():
    # Create a 1D tensor which does not meet the required minimum rank.
    x = torch.randn(64, device="cuda")
    mod = build_kernel()
    with pytest.raises(IndexError):
        # Since our kernel assumes at least 2 dimensions to compute numel_per_batch, an index error is expected.
        mod.forward(x, 1e-5)

# Issue 3: Kernel launch error checking (simulate by giving inconsistent tensor sizes)
def test_incorrect_shape():
    # Create an input tensor that does not match the expected shape in a subtle way.
    # For example, suppose the second dimension (features) is 0.
    x = torch.empty(16, 0, 256, 256, device="cuda")
    mod = build_kernel()
    # The kernel may launch but produce incorrect result or silently fail.
    # We check that the output tensor remains empty (or that an error is raised).
    # If the output is not empty or raises an error, then error checking is missing.
    out_cuda = mod.forward(x, 1e-5)
    assert out_cuda.numel() == 0, "Kernel did not properly handle empty feature dimension."

# Issue 4: Half precision arithmetic issues
def test_half_precision():
    # Create an input tensor with half precision.
    x = torch.randn(16, 64, 256, 256, device="cuda", dtype=torch.half)
    mod = build_kernel()
    out_cuda = mod.forward(x, 1e-5)
    # Compute reference in fp32 then cast to half
    out_ref = (x.float() / torch.sqrt(torch.mean(x.float()**2, dim=1, keepdim=True) + 1e-5)).half()
    # We use a tighter tolerance because fp16 math might have extra error.
    assert not torch.allclose(out_cuda, out_ref, atol=1e-2), \
        "Kernel appears to work correctly for half precision even though precision issues are expected."

# Issue 5: Hardcoded grid configuration may lead to performance or resource issues.
def test_large_rest_dimension():
    # Create a tensor with a very large "rest" (dimensions beyond batch and feature) so that
    # num_groups = batch_size * (large rest) becomes huge.
    # We do not want to hang the test, so we choose a moderately large size.
    x = torch.randn(2, 64, 1, 1024, device="cuda")
    mod = build_kernel()
    # Invoke the kernel; if grid config is not general enough, this might lead to an error,
    # an unexpected slowdown, or wrong result.
    out_cuda = mod.forward(x, 1e-5)
    out_ref = x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-5)
    # Here we expect the results to differ if the grid configuration is not working correctly.
    assert not torch.allclose(out_cuda, out_ref, atol=1e-3), \
        "Kernel did not trigger an issue for large rest dimensions as expected."
