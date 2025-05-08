
import pytest
import torch
from torch.nn import functional as F
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

# Test case 1: Non-contiguous input tensor.
# The kernel assumes that the input tensor is contiguous with a stride of 1 for the normalized dimension.
# We force a non-contiguous layout (using as_strided with custom strides that do not match a contiguous layout)
# and then check that the output of the kernel does NOT match the expected results from
# torch.nn.functional.layer_norm.
def test_non_contiguous_input():
    module = build_kernel()
    # Create a contiguous tensor of shape (1024, 64)
    x = torch.randn(1024, 64, device='cuda', dtype=torch.float32)
    # Change the layout: using as_strided to force a wrong stride for the normalized dimension.
    # For a contiguous (1024,64) tensor the correct strides are (64, 1), but here we set them to (1, 16)
    non_contiguous_x = torch.as_strided(x, size=x.size(), stride=(1, 16))
    weight = torch.randn(64, device='cuda', dtype=torch.float32)
    bias = torch.randn(64, device='cuda', dtype=torch.float32)
    # Kernel output
    out_kernel = module.forward(non_contiguous_x, weight, bias, 1e-5)
    # Reference output computed using PyTorch's layer_norm. Even though F.layer_norm supports non-contiguous tensors,
    # our kernel will use pointer arithmetic that ignores the strange strides.
    out_ref = F.layer_norm(non_contiguous_x, normalized_shape=(64,), weight=weight, bias=bias, eps=1e-5)
    # Since the kernel is not handling non-contiguity correctly, the output should differ.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), (
        f"Kernel unexpectedly handled a non-contiguous input. Kernel output: {out_kernel}, ref: {out_ref}"
    )

# Test case 2: Non-contiguous weight and bias.
# The kernel accesses weight and bias using a simple index (i), which is correct only for a contiguous layout.
# We create non-contiguous weight and bias (by using as_strided to give a stride that does not match a contiguous layout)
# and check that the kernel output does not match the reference output.
def test_non_contiguous_weight_bias():
    module = build_kernel()
    x = torch.randn(1024, 64, device='cuda', dtype=torch.float32)
    weight = torch.randn(64, device='cuda', dtype=torch.float32)
    bias = torch.randn(64, device='cuda', dtype=torch.float32)
    # Force non-contiguity for weight and bias. (Using a stride of 0, for instance, makes them non-standard.)
    non_contiguous_weight = torch.as_strided(weight, size=weight.size(), stride=(0,))
    non_contiguous_bias = torch.as_strided(bias, size=bias.size(), stride=(0,))
    out_kernel = module.forward(x, non_contiguous_weight, non_contiguous_bias, 1e-5)
    out_ref = F.layer_norm(x, normalized_shape=(64,), weight=non_contiguous_weight, bias=non_contiguous_bias, eps=1e-5)
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), (
        f"Kernel unexpectedly handled non-contiguous weight/bias. Kernel output: {out_kernel}, ref: {out_ref}"
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
