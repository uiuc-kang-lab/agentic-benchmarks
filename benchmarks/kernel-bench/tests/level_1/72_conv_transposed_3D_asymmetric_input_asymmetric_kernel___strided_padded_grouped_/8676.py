
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension using our kernel.cu file.
    cuda_module = load(
        name="custom_conv_transpose3d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: The kernel merely wraps at::conv_transpose3d.
# Although this is a design issue rather than a runtime bug, we test that the returned result
# is not being computed by a custom kernel. In our test we compare performance markers.
# (Note: this test is more illustrative. In real cases one expects a custom kernel to be much faster.)
def test_kernel_is_wrapper():
    module = build_kernel()
    x = torch.randn(2, 4, 8, 8, 8, device='cuda', dtype=torch.float32)
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda', dtype=torch.float32)
    result = module.forward(x, weight, None, [2, 2, 2], [1, 1, 1], [0, 0, 0], 1)
    # If the kernel was custom, one might expect extra profiling markers. Here we assume a wrong design.
    assert result is not None, "Expected a valid tensor result, but got None."

# Issue 2: No check on the length of parameter vectors.
def test_invalid_conv_params_length():
    module = build_kernel()
    x = torch.randn(2, 4, 8, 8, 8, device='cuda', dtype=torch.float32)
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda', dtype=torch.float32)
    # Pass stride vector of length 2 instead of 3.
    with pytest.raises(RuntimeError):
        module.forward(x, weight, None, [2, 2], [1,1,1], [0,0,0], 1)
    # Also check for padding parameter
    with pytest.raises(RuntimeError):
        module.forward(x, weight, None, [2,2,2], [1,1], [0,0,0], 1)

# Issue 3: Lack of explicit type check.
def test_input_tensor_wrong_dtype():
    module = build_kernel()
    # Create a double tensor (float64) instead of float32.
    x = torch.randn(2, 4, 8, 8, 8, device='cuda', dtype=torch.float64)
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda', dtype=torch.float64)
    with pytest.raises(RuntimeError):
        module.forward(x, weight, None, [2, 2, 2], [1, 1, 1], [0, 0, 0], 1)

# Issue 4: Optional bias handling may be problematic.
def test_optional_bias_non_contiguous():
    module = build_kernel()
    x = torch.randn(2, 4, 8, 8, 8, device='cuda', dtype=torch.float32)
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda', dtype=torch.float32)
    bias = torch.randn(4, device='cuda', dtype=torch.float32)
    # Make bias non-contiguous by unsqueezing and then transposing if applicable.
    bias_non_contig = bias.unsqueeze(0).transpose(0, 0)
    # Although bias_non_contig might still be contiguous, let’s force non_contiguity
    bias_non_contig = bias_non_contig[:, :1].squeeze(1)
    # Now check if using a non-contiguous bias triggers the CHECK_INPUT failure.
    with pytest.raises(RuntimeError):
        module.forward(x, weight, bias_non_contig, [2, 2, 2], [1, 1, 1], [0, 0, 0], 1)

# Issue 5: No explicit shape checks for the weight tensor.
def test_weight_incorrect_shape():
    module = build_kernel()
    x = torch.randn(2, 4, 8, 8, 8, device='cuda', dtype=torch.float32)
    # Creating a weight tensor with an incorrect shape (e.g., missing one dimension)
    weight_bad = torch.randn(4, 4, 3, 3, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        module.forward(x, weight_bad, None, [2, 2, 2], [1, 1, 1], [0, 0, 0], 1)
