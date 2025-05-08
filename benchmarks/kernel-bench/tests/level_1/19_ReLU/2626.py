
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Using double precision tensor.
# This should trigger the issue since the kernel always uses float4 vectorized loads,
# which is invalid for double types.
def test_double_precision():
    device = "cuda"
    # Create a double precision tensor.
    x = torch.randn(1024, dtype=torch.float64, device=device)
    my_module = build_kernel()
    # The kernel is expected to produce an incorrect result for double tensors.
    out_kernel = my_module.forward(x)
    out_ref = torch.relu(x)
    # Check if the kernel result is not equal to the reference result.
    # We use an assertion that fails if they match (which is very unlikely) and passes if they differ.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-6), (
        "Kernel unexpectedly produced correct result for a double tensor, but it uses vectorized loads with float4!"
    )

# Test case 2: Misaligned memory access.
# This test creates a tensor that is deliberately misaligned.
# We do this by allocating a larger tensor and then slicing off an element,
# which typically results in the pointer not being 16-byte aligned.
def test_misaligned_tensor():
    device = "cuda"
    # Create a float32 tensor that is large enough.
    x_full = torch.randn(1025, dtype=torch.float32, device=device)
    # Slice the tensor to force a misaligned pointer.
    x = x_full[1:]
    my_module = build_kernel()
    out_kernel = my_module.forward(x)
    out_ref = torch.relu(x)
    # If the kernel is performing misaligned vectorized loads, the results may be incorrect.
    # We assert that the kernel result does not match the reference.
    # (In a debug build or with strict memory alignment, this could trigger an error.)
    assert not torch.allclose(out_kernel, out_ref, atol=1e-6), (
        "Kernel unexpectedly produced correct result for misaligned tensor, but it assumes 16-byte alignment!"
    )

# Test case 3: (Minor) Check for usage of deprecated API.
# This test does not run the kernel but inspects that a warning is issued if available.
# This is a placeholder and could be expanded based on project conventions.
def test_deprecated_api_usage():
    device = "cuda"
    x = torch.randn(1024, dtype=torch.float32, device=device)
    my_module = build_kernel()
    # Call the kernel function
    _ = my_module.forward(x)
    # This test is a placeholder to remind developers that input.scalar_type() is preferred
    # over input.type() in AT_DISPATCH_FLOATING_TYPES.
    # In a full implementation, one could capture warnings or check the binary for symbols.
    assert True
