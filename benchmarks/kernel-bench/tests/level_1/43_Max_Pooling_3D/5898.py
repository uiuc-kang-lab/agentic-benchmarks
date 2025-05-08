
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="max_pool3d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Misleading comment about shared memory usage.
# Although we cannot directly inspect shared memory use from Python,
# we can at least compare the kernel’s output against PyTorch’s built-in MaxPool3d.
def test_issue_shared_memory():
    # Use a moderately large tensor to exercise the kernel.
    input_tensor = torch.randn(64, 16, 32, 32, 32, device="cuda")
    model = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1)
    ref_out = model(input_tensor)
    
    kernel_module = build_kernel()
    kernel_out = kernel_module.forward(input_tensor, 3, 2, 1, 1, False, False)
    torch.cuda.synchronize()
    
    # If output is incorrect, it might indicate that the omission of shared memory
    # (and any potential performance-related issues) has affected the result.
    assert torch.allclose(kernel_out, ref_out, atol=1e-5), (
        "Kernel output does not match PyTorch MaxPool3d output. "
        "This may be due to the missing shared memory optimization."
    )

# Issue 2: Kernel assumption of contiguous input.
def test_issue_non_contiguous():
    # Create a contiguous input tensor and then make it non-contiguous via transpose.
    input_tensor = torch.randn(16, 32, 64, 64, 64, device="cuda")
    non_contiguous = input_tensor.transpose(2, 3)  # Now non-contiguous

    kernel_module = build_kernel()
    out_noncontig = kernel_module.forward(non_contiguous, 3, 2, 1, 1, False, False)
    out_contig = kernel_module.forward(non_contiguous.contiguous(), 3, 2, 1, 1, False, False)
    torch.cuda.synchronize()
    
    assert torch.allclose(out_noncontig, out_contig, atol=1e-5), (
        "Kernel output differs between non-contiguous and contiguous inputs."
    )

# Issue 3: Lack of support for half precision.
def test_issue_half_precision():
    # Create a half precision input tensor.
    input_tensor = torch.randn(16, 32, 64, 64, 64, device="cuda", dtype=torch.half)
    kernel_module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # Expecting an error because half is not dispatched in AT_DISPATCH_FLOATING_TYPES.
        _ = kernel_module.forward(input_tensor, 3, 2, 1, 1, False, False)

# Issue 4: Incompatible format when return_indices is true.
def test_issue_return_indices_format():
    # Create a standard input tensor.
    input_tensor = torch.randn(16, 32, 64, 64, 64, device="cuda")
    kernel_module = build_kernel()
    result = kernel_module.forward(input_tensor, 3, 2, 1, 1, True, False)
    torch.cuda.synchronize()
    
    # Standard PyTorch MaxPool3d returns a tuple (output, indices) when return_indices is true.
    # Our kernel returns a single stacked tensor with an extra dimension at dim0.
    # Therefore we expect the result to be a 6D tensor where dim0 has size 2.
    assert result.dim() == 6 and result.size(0) == 2, (
        "Return indices output format is not a stacked tensor with dim0==2. "
        "This deviates from expected behavior."
    )
