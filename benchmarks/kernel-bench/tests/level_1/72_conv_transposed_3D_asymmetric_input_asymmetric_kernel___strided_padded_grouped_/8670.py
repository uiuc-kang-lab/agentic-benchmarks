
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility to compile and load the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose3d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# 1. Test non-implementation of a custom kernel by triggering a scenario where
#    timing differences or special CUDA grid/block behaviors might be expected.
#    While we cannot force custom behavior, we can at least note that the wrapper
#    simply defers to at::conv_transpose3d.
def test_wrapper_vs_custom_kernel():
    mod = build_kernel()
    # Create valid contiguous inputs.
    x = torch.randn(2, 4, 8, 8, 8, device='cuda')
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda')
    # No bias provided.
    # This test essentially ensures that the wrapper does not perform any custom logic.
    out = mod.forward(x, weight, None, [1, 1, 1], [0, 0, 0], [0, 0, 0], 1)
    # We assume that if the custom kernel was implemented, we would see specialized CUDA launches.
    # Here, we only check that output has expected dimensions.
    assert out.shape[2] == (x.shape[2] - 1) * 1 - 2*0 + 3, "Output depth is not as expected"

# 2. Test that non-contiguous input tensors trigger an error.
def test_non_contiguous_input():
    mod = build_kernel()
    # Create non-contiguous input by transposing one of the spatial dimensions.
    x = torch.randn(2, 4, 8, 8, 8, device='cuda')
    x_noncontig = x.transpose(2, 3)  # likely to be non-contiguous
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda')
    with pytest.raises(RuntimeError, match="must be contiguous"):
        mod.forward(x_noncontig, weight, None, [1,1,1], [0,0,0], [0,0,0], 1)

# 3. Test that a tensor of an unexpected data type (non float32) triggers an error.
def test_wrong_dtype():
    mod = build_kernel()
    # Create input and weight tensors in double precision.
    x = torch.randn(2, 4, 8, 8, 8, device='cuda', dtype=torch.double)
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda', dtype=torch.double)
    # The CHECK_INPUT macros in our kernel do not check dtype, but underlying at::conv_transpose3d
    # may not support double for CUDA if not registered. So we expect an error or a type-cast issue.
    with pytest.raises(RuntimeError):
        mod.forward(x, weight, None, [1,1,1], [0,0,0], [0,0,0], 1)

# 4. Test that a shape mismatch in the bias tensor is caught.
def test_bias_shape_validation():
    mod = build_kernel()
    x = torch.randn(2, 4, 8, 8, 8, device='cuda')
    weight = torch.randn(4, 4, 3, 3, 3, device='cuda')
    # Create a bias tensor with an incorrect shape.
    wrong_bias = torch.randn(10, device='cuda')  # expected shape should match out_channels (or per channel)
    with pytest.raises(RuntimeError):
        mod.forward(x, weight, wrong_bias, [1,1,1], [0,0,0], [0,0,0], 1)
        
if __name__ == '__main__':
    pytest.main()
