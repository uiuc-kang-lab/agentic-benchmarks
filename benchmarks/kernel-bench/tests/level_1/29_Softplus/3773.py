
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="softplus_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def compute_softplus(x: torch.Tensor) -> torch.Tensor:
    # Reference softplus using PyTorch's functional implementation
    return torch.nn.functional.softplus(x)

# Issue 1: Test that passing a half precision tensor raises an error or gives incorrect result.
def test_half_precision_not_supported():
    kernel_module = build_kernel()
    # Create a half precision tensor on CUDA.
    # Our kernel dispatch macros don't cover half so we expect a RuntimeError or unexpected behavior.
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # The launch should fail because half is not supported by our AT_DISPATCH_FLOATING_TYPES.
        y = kernel_module.forward(x)
        # synchronize to force error propagation if asynchronous
        torch.cuda.synchronize()

# Issue 2: Test that non-contiguous inputs produce incorrect results.
def test_non_contiguous_input():
    kernel_module = build_kernel()
    # create a contiguous tensor then make it non-contiguous via transposition.
    x = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    x_noncontig = x.transpose(0, 1)  # this makes the tensor non-contiguous
    # Reference softplus computed using torchvision function on noncontiguous tensor
    ref = compute_softplus(x_noncontig)
    # Call our custom kernel. Since the kernel simply calls data_ptr() and iterates
    # over assumed contiguous memory, it will process the wrong memory layout.
    y = kernel_module.forward(x_noncontig)
    torch.cuda.synchronize()
    # The outputs will not match because of non-contiguity: We expect a difference.
    assert not torch.allclose(y, ref, atol=1e-5), (
        "Expected the kernel to produce an incorrect result for non-contiguous input."
    )

# Issue 3: Test that passing a CPU tensor (when a CUDA tensor is expected) triggers an error.
def test_cpu_input_error():
    kernel_module = build_kernel()
    x = torch.randn(1024, dtype=torch.float32)  # CPU tensor
    with pytest.raises(RuntimeError):
        # The kernel is written for CUDA so passing a CPU tensor should raise an error.
        y = kernel_module.forward(x)
        torch.cuda.synchronize()
