
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="selu_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Non-contiguous tensor input
# This test creates a non-contiguous tensor (by transposing) and passes it to the kernel.
# Because the kernel assumes contiguous memory, its output will be incorrect.
def test_non_contiguous_input():
    kernel = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous via a transpose
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x_non_contiguous = x.t()  # transpose makes it non-contiguous
    # Note: The reference computation (torch.selu) works correctly with non-contiguous tensors.
    output_kernel = kernel.forward(x_non_contiguous)
    output_ref = torch.selu(x_non_contiguous)
    # They are expected to differ because the kernel uses a simple linear indexing.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), (
        "Kernel should produce incorrect results on non-contiguous inputs, but it did not!"
    )

# Test 2: Half precision (float16) input
# This test checks that passing a half precision tensor, which is currently not supported by the kernel,
# triggers an error.
def test_half_precision_input():
    kernel = build_kernel()
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # The AT_DISPATCH_FLOATING_TYPES macro does not cover half precision.
        _ = kernel.forward(x)

# Test 3: Dummy warp-level reduction overhead (optional)
# This test aims to ensure that the extra warp-level reduction code does not accidentally change the SELU output.
# Even though the dummy reduction is not expected to alter results (apart from performance overhead),
# we verify that the computed activation is mathematically identical to torch.selu when using contiguous input.
def test_warp_reduction_correctness():
    kernel = build_kernel()
    x = torch.randn(2048, device="cuda", dtype=torch.float32)
    output_kernel = kernel.forward(x)
    output_ref = torch.selu(x)
    # They should be equal as the dummy warp reduction multiplies by one.
    assert torch.allclose(output_kernel, output_ref, atol=1e-5), (
        "Kernel output (with warp-level reduction) does not match expected SELU activation output."
    )
