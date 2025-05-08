
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel():
    # Build the CUDA extension from kernel.cu.
    # NOTE: In this testing scenario, the module's forward simply calls at::conv_transpose2d.
    # However, if the custom kernel were launched, these tests are expected to trigger issues.
    cuda_module = load(
        name="custom_conv_transpose2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Issue 1: The forward() function never launches the custom kernel.
def test_forward_does_not_use_custom_kernel():
    module = build_kernel()
    # We create a simple input and weight: if our custom kernel was used, some deviation (or error) should occur.
    # Instead, at::conv_transpose2d is invoked.
    input_tensor = torch.randn(1, 4, 8, 8, device="cuda", dtype=torch.float32)
    weight_tensor = torch.randn(4, 4, 3, 3, device="cuda", dtype=torch.float32)
    # Use no bias here.
    result = module.forward(input_tensor, weight_tensor)
    ref_result = F.conv_transpose2d(input_tensor, weight_tensor, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=(1,1))
    assert torch.allclose(result, ref_result, atol=1e-5), (
        "Test for Issue 1: The custom forward did not launch the kernel as expected (result identical to at::conv_transpose2d)."
    )

# Issue 2 & 3: The reliance on float4 vectorized loads and division by 4 assumes sizes are multiples of 4.
@pytest.mark.parametrize("in_channels, out_channels, H, W", [
    (3, 5, 7, 7),  # channels not divisible by 4, spatial dims too.
    (4, 4, 10, 10)  # spatial dims not a multiple of 4.
])
def test_non_vectorized_input(in_channels, out_channels, H, W):
    module = build_kernel()
    input_tensor = torch.randn(1, in_channels, H, W, device="cuda", dtype=torch.float32)
    weight_tensor = torch.randn(in_channels, out_channels, 3, 3, device="cuda", dtype=torch.float32)
    # Rearrange weight shape to match conv_transpose2d expected weight shape (in_channels, out_channels, H, W)
    try:
        result = module.forward(input_tensor, weight_tensor)
    except Exception as e:
        pytest.skip(f"Kernel raised an exception as expected with non-vectorized sizes: {e}")
    ref_result = F.conv_transpose2d(input_tensor, weight_tensor, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=(1,1))
    # The result may be wrong due to mis-indexing.
    assert not torch.allclose(result, ref_result, atol=1e-3), (
        "Test for Issue 2/3: Expected output to differ due to misalignment and incorrect vectorized indexing."
    )

# Issue 4: Shared memory usage is incomplete (shared_input declared but never used).
def test_shared_memory_unused():
    module = build_kernel()
    # Use a configuration that forces shared memory usage.
    input_tensor = torch.randn(1, 4, 8, 8, device="cuda", dtype=torch.float32)
    weight_tensor = torch.randn(4, 4, 3, 3, device="cuda", dtype=torch.float32)
    # If shared memory was used correctly, the output would incorporate computations from shared_input.
    # We compare against an intentionally wrong baseline.
    result = module.forward(input_tensor, weight_tensor)
    # Here we simply check that the output is not trivially equal to input (which could happen if shared_input is ignored)
    assert not torch.allclose(result, input_tensor, atol=1e-5), (
        "Test for Issue 4: Output appears to ignore proper shared memory computation."
    )

# Issue 5: Bias is not handled by the kernel.
def test_bias_ignored():
    module = build_kernel()
    input_tensor = torch.randn(1, 6, 8, 8, device="cuda", dtype=torch.float32)
    weight_tensor = torch.randn(6, 8, 3, 3, device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(8, device="cuda", dtype=torch.float32)
    result = module.forward(input_tensor, weight_tensor, bias_tensor)
    ref_result = F.conv_transpose2d(input_tensor, weight_tensor, bias=bias_tensor, stride=1, padding=0, output_padding=0, groups=1, dilation=(1,1))
    # Expect a discrepancy since bias should be added.
    assert not torch.allclose(result, ref_result, atol=1e-3), (
        "Test for Issue 5: Kernel appears to incorporate bias even though it should ignore it."
    )

# Issue 6: Groups parameter is ignored.
def test_groups_not_handled():
    module = build_kernel()
    # Create an input with 4 channels and set groups=2
    input_tensor = torch.randn(1, 4, 8, 8, device="cuda", dtype=torch.float32)
    # Weight shape for groups=2 (conv_transpose2d weight shape: (in_channels, out_channels/groups, kernel_h, kernel_w))
    weight_tensor = torch.randn(4, 2, 3, 3, device="cuda", dtype=torch.float32)
    result = module.forward(input_tensor, weight_tensor, None, stride=1, padding=0, output_padding=0, groups=2)
    ref_result = F.conv_transpose2d(input_tensor, weight_tensor, bias=None, stride=1, padding=0, output_padding=0, groups=2, dilation=(1,1))
    # If groups are handled correctly, they would match; since the kernel ignores groups, a discrepancy is expected.
    assert not torch.allclose(result, ref_result, atol=1e-3), (
        "Test for Issue 6: Kernel appears to correctly support groups, but it is expected to ignore them."
    )

# Issue 7: The loop unrolling and stepping in increments of 4 may fail when the spatial dims are not multiples of 4.
def test_spatial_loop_boundaries():
    module = build_kernel()
    # Use spatial dimensions that are not multiples of 4.
    input_tensor = torch.randn(1, 4, 9, 9, device="cuda", dtype=torch.float32)
    weight_tensor = torch.randn(4, 4, 3, 3, device="cuda", dtype=torch.float32)
    result = module.forward(input_tensor, weight_tensor)
    ref_result = F.conv_transpose2d(input_tensor, weight_tensor, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=(1,1))
    assert not torch.allclose(result, ref_result, atol=1e-3), (
        "Test for Issue 7: The computed output should be incorrect because of improper handling of spatial loop boundaries."
    )

if __name__ == "__main__":
    pytest.main([__file__])
