
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="swish_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper: reference swish activation computed using PyTorch.
def swish_torch(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

# Test 1: Wrong dtype (non-float32) should produce wrong results.
def test_wrong_dtype():
    cuda_module = build_kernel()
    # Create a float64 tensor on GPU.
    x = torch.randn(1024, 1024, dtype=torch.float64, device='cuda')
    # Although our kernel doesn't check dtype, it will misinterpret the double data.
    y_kernel = cuda_module.forward(x)
    # Use PyTorch's swish as reference (computed in correct dtype).
    y_ref = swish_torch(x)
    # They should not match due to wrong interpretation.
    assert not torch.allclose(y_kernel.to(torch.float64), y_ref, atol=1e-5), \
        "Kernel incorrectly handled non-float32 input."

# Test 2: Non-contiguous input tensor should lead to an incorrect output.
def test_non_contiguous():
    cuda_module = build_kernel()
    # Create a contiguous tensor and then transpose it to be non-contiguous.
    x = torch.randn(256, 1024, device='cuda', dtype=torch.float32)
    x_non_contig = x.t()  # transposed -> non-contiguous view.
    # Make sure x_non_contig is non-contiguous.
    assert not x_non_contig.is_contiguous(), "Test setup error: x_non_contig should be non-contiguous."

    y_kernel = cuda_module.forward(x_non_contig)
    y_ref = swish_torch(x_non_contig)
    # The outputs should differ because the kernel incorrectly assumes contiguity.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), \
        "Kernel incorrectly handled non-contiguous input."

# Test 3: Passing a CPU tensor should raise an error because the kernel expects a CUDA tensor.
def test_cpu_input():
    cuda_module = build_kernel()
    x_cpu = torch.randn(1024, 1024, device='cpu', dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input tensor must be on CUDA"):
        # The TORCH_CHECK in the kernel should trigger an error.
        _ = cuda_module.forward(x_cpu)

# Test 4: (Index overflow issue) This test simulates the potential issue.
# Since we cannot really allocate > INT_MAX elements on the GPU, we simulate by patching the kernel call.
# Here we check that for a moderately large tensor (but within safe limits) the kernel still matches swish.
# Note: This test will pass even though the int versus int64_t issue exists, because our test tensor is small.
def test_large_tensor_behavior():
    cuda_module = build_kernel()
    # This tensor is large but within safe limits.
    # Note: In real cases, an extremely large tensor (n > INT_MAX) might trigger index overflow.
    n = 1 << 20  # about one million elements, safely below INT_MAX.
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    y_kernel = cuda_module.forward(x)
    y_ref = swish_torch(x)
    assert torch.allclose(y_kernel, y_ref, atol=1e-5), \
        "Kernel did not compute correct swish activation on a large tensor (within safe limits)."
