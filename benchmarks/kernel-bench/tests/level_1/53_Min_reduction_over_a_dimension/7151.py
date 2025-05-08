
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Helper: compare our kernel result with PyTorch reference (for valid cases)
def reference_min(input, dim):
    return torch.min(input, dim=dim)[0]

# Issue 1: Use of std::numeric_limits with half precision (__half) may be unsupported.
def test_half_precision():
    kernel_module = build_kernel()
    # Create a half precision tensor. Many kernels using std::numeric_limits may not work with __half.
    x = torch.randn(8, 32, 16, device="cuda", dtype=torch.half)
    # Expect the kernel to either error out or produce incorrect result.
    with pytest.raises(RuntimeError):
        _ = kernel_module.forward(x, 1)
        
# Issue 2: Shared memory size inconsistent (unused variable computed with sizeof(float) vs. sizeof(scalar_t)).
# For a non-float32 type such as double the kernel launch uses sizeof(double),
# but the presence of the redundant variable might hint at overlooked bugs.
def test_double_precision():
    kernel_module = build_kernel()
    x = torch.randn(8, 32, 16, device="cuda", dtype=torch.double)
    # Run kernel. In a correct implementation, kernel output should match torch.min reduction.
    y_kernel = kernel_module.forward(x, 1)
    y_ref = reference_min(x, 1)
    # If there is a shared memory miscalculation, the result might be off.
    assert torch.allclose(y_kernel, y_ref, atol=1e-6), f"Double precision reduction failed. max diff {(y_kernel - y_ref).abs().max()}"

# Issue 3: Lack of kernel launch error checking.
# Trigger an error by passing an invalid dimension. The host function does check for dim in [0, ndim),
# so we pass an out-of-range dimension to trigger a TORCH_CHECK failure.
def test_invalid_dimension():
    kernel_module = build_kernel()
    x = torch.randn(8, 32, 16, device="cuda")
    with pytest.raises(RuntimeError):
        _ = kernel_module.forward(x, 5)  # Invalid: input.dim() == 3, index 5 out-of-range

# Issue 4: Kernel assumes the input tensor is contiguous and has the expected layout.
# Here we create a noncontiguous tensor by transposing a contiguous tensor.
def test_non_contiguous_input():
    kernel_module = build_kernel()
    x = torch.randn(8, 32, 16, device="cuda")
    # Make tensor noncontiguous via a transpose.
    x_nc = x.transpose(0, 1)
    # The kernel forces contiguity, but this may change the expected memory layout.
    # Therefore, the kernel’s reduction (which assumes a specific memory ordering)
    # might produce an incorrect result compared to torch.min.
    y_kernel = kernel_module.forward(x_nc, 1)
    y_ref = reference_min(x_nc.contiguous(), 1)
    # Since the kernel computes reduction using a specific contiguous layout,
    # the noncontiguous input may result in wrong output.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), "Kernel unexpectedly handled noncontiguous input correctly."

# Issue 5: Kernel does not handle the empty reduction dimension (r==0)
def test_empty_reduction_dimension():
    kernel_module = build_kernel()
    # Create a tensor where the reduction dimension has size 0.
    x = torch.randn(4, 0, 10, device="cuda")
    # Depending on implementation, either an error is raised or an undefined result is produced.
    with pytest.raises(RuntimeError):
        _ = kernel_module.forward(x, 1)

if __name__ == "__main__":
    pytest.main([__file__])
