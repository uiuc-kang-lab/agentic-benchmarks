
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension in place. We assume kernel.cu is in the same directory.
def build_kernel():
    module = load(
        name="hardtanh_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Test 1: Trigger misaligned memory access by creating a non-contiguous tensor
def test_misaligned_tensor():
    my_module = build_kernel()
    # Create a tensor with an extra element then take a slice that is likely misaligned.
    # For float32, a tensor starting at offset 1 will have its underlying data misaligned to 4 bytes off from a 16-byte boundary.
    base = torch.randn(1025, device="cuda", dtype=torch.float32)
    x = base[1:]  # Not contiguous or misaligned pointer likely.
    # Save reference using torch.clamp to simulate F.hardtanh
    expected = torch.clamp(x, min=-1.0, max=1.0)
    out = my_module.forward(x, -1.0, 1.0)
    torch.cuda.synchronize()
    # We expect that even if misalignment issues exist, the output will be wrong.
    # Thus, this test fails if outputs do not match.
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), "Output does not match expected clamp for misaligned tensor"

# Test 2: Test using a custom CUDA stream (the kernel creates its own stream)
def test_kernel_uses_new_stream():
    my_module = build_kernel()
    # Create a moderately sized tensor.
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    expected = torch.clamp(x, min=-1.0, max=1.0)
    # Call the kernel, which internally creates and synchronizes its own stream.
    out = my_module.forward(x, -1.0, 1.0)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), "Kernel execution with new stream failed to produce expected result"

# Test 3: Trigger the clamped grid dimension by creating a very large tensor.
def test_large_tensor_grid_clamp():
    my_module = build_kernel()
    # Create an input tensor larger than 65,535 * 256 elements.
    num_elements = (65535 + 1) * 256
    x = torch.randn(num_elements, device="cuda", dtype=torch.float32)
    expected = torch.clamp(x, min=-1.0, max=1.0)
    out = my_module.forward(x, -1.0, 1.0)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), "Kernel did not process a large tensor correctly under grid block clamp"

# Test 4: Pass a non-contiguous tensor to trigger potential issues with vectorized loads.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a 2D tensor then transpose, which makes it non-contiguous.
    x = torch.randn(32, 128, device="cuda", dtype=torch.float32).t()
    expected = torch.clamp(x, min=-1.0, max=1.0)
    out = my_module.forward(x, -1.0, 1.0)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), "Kernel did not correctly process a non-contiguous tensor"

# Test 5: Passing an integer tensor should raise an error (or produce an unexpected result) due to the AT_DISPATCH_FLOATING_TYPES.
def test_integer_tensor_error():
    my_module = build_kernel()
    x = torch.randint(-10, 10, (256,), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        # Expect an error because kernel dispatch only supports floating point types.
        _ = my_module.forward(x, -1.0, 1.0)

# Test 6: Passing a CPU tensor should raise an error.
def test_cpu_tensor_error():
    my_module = build_kernel()
    x = torch.randn(256, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = my_module.forward(x, -1.0, 1.0)
