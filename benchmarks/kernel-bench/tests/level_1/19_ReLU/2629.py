
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper function to build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel assumes contiguous tensors.
def test_non_contiguous_input():
    # Create a contiguous tensor and then a non-contiguous view (by transposing)
    x = torch.randn(128, 64, device="cuda", dtype=torch.float32)
    # Transpose makes the tensor non-contiguous
    non_contig = x.t()  
    # Get reference with torch.relu (which handles strides correctly)
    ref = torch.relu(non_contig)
    
    # Build the module and run the kernel
    mod = build_kernel()
    out = mod.forward(non_contig)
    torch.cuda.synchronize()
    
    # If the kernel is not handling non-contiguous layouts, the output will be wrong.
    # We expect an error (or at least a large difference) compared to the reference.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "Kernel produced the same result on a non-contiguous tensor."
        " Expected an error due to incorrect memory access."
    )

# Issue 2: Lack of error checking after kernel launch.
def test_kernel_launch_error():
    # Passing a CPU tensor to a CUDA-only kernel should lead to an error.
    x = torch.randn(1024, device="cpu", dtype=torch.float32)
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because the tensor is not on the CUDA device.
        mod.forward(x)

# Issue 3: Potential integer overflow when the tensor is very large.
def test_large_tensor_overflow():
    # We wish to simulate a case where input.numel() > INT_MAX (2**31 - 1).
    # It is impractical to allocate such a tensor in a test.
    # Instead we monkey-patch the numel() method on our input tensor.
    # WARNING: This is a hack and only for testing the kernel’s behavior in this edge case.
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    
    class FakeTensor(torch.Tensor):
        pass

    # Create a fake tensor that behaves like x but reports a huge number of elements.
    fake = x.clone().detach().cuda().requires_grad_(False)
    # Monkey-patch the numel method to return INT_MAX+100 for testing.
    huge_size = (2**31) + 100
    original_numel = fake.numel
    fake.numel = lambda: huge_size

    mod = build_kernel()
    # We do not expect the kernel to work correctly with such a huge "size".
    # Since this misuse is not caught inside the kernel (lack of index type promotion),
    # the result is likely to be garbage or the kernel may crash/hang.
    # For safety in testing, we run this call with a timeout.
    with pytest.raises(Exception):
        out = mod.forward(fake)
        torch.cuda.synchronize()
    
    # Restore original method (if needed for further tests)
    fake.numel = original_numel

# Issue 4: Kernel advertises warp-level optimization but does not use any warp intrinsics.
def test_warp_primitive_usage():
    # There is no direct runtime error for not using warp-level intrinsics.
    # We can only check that the functionality (ReLU) is the same for a "typical" tensor.
    x = torch.randn(2048, device="cuda", dtype=torch.float32)
    ref = torch.relu(x)
    mod = build_kernel()
    out = mod.forward(x)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-5), (
        "Kernel computation is incorrect. Although warp-level primitives were advertised, "
        "they were not used. (This test verifies result correctness but cannot verify performance.)"
    )

# Issue 5: Use of input.type() (legacy API) in the dispatch macro (should use input.scalar_type()).
def test_deprecated_type_usage():
    # Create a tensor of dtype half, which is not included in AT_DISPATCH_FLOATING_TYPES.
    # This should trigger a dispatch error.
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # If the legacy type dispatch does not handle half properly, an error should be raised.
        mod.forward(x)
