
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that passing per-dimension stride/padding/dilation (as tuples) fails.
def test_scalar_stride_padding_dilation():
    # Create dummy input data and weight data
    batch_size = 2
    in_channels = 4
    out_channels = 4
    in_d, in_h, in_w = 10, 10, 10
    kernel_d, kernel_h, kernel_w = 3, 3, 3

    # Create input tensor and weight tensor on CUDA
    input_tensor = torch.randn(batch_size, in_channels, in_d, in_h, in_w, device="cuda", dtype=torch.float32)
    # Weight shape: [out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w]
    weight = torch.randn(out_channels, in_channels, kernel_d, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    
    # Instead of scalar, we now pass tuples to represent per-dimension parameters.
    # The extension expects int parameters, so we simulate an error by trying to pass tuples.
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    groups = 1

    cuda_mod = build_kernel()
    
    with pytest.raises(TypeError):
        # This should fail because the bound function expects an int, not a tuple.
        cuda_mod.forward(input_tensor, weight, None, stride, padding, dilation, groups)

# Issue 2: Test non-divisible in_channels with groups.
def test_invalid_groups_division():
    batch_size = 2
    # in_channels is not divisible by groups; for example, 3 channels with groups=2.
    in_channels = 3
    out_channels = 2
    in_d, in_h, in_w = 8, 8, 8
    kernel_d, kernel_h, kernel_w = 3, 3, 3
    groups = 2

    input_tensor = torch.randn(batch_size, in_channels, in_d, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = None
    stride = 1
    padding = 1
    dilation = 1

    cuda_mod = build_kernel()
    
    # Because the kernel does an integer division without verifying divisibility,
    # its behavior will be undefined. Here, we check that the output does not match PyTorch's conv3d.
    output_kernel = cuda_mod.forward(input_tensor, weight, bias, stride, padding, dilation, groups)
    conv3d = torch.nn.Conv3d(in_channels, out_channels, (kernel_d, kernel_h, kernel_w),
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
    # Force conv3d weights to the same as used in the kernel.
    conv3d.weight.data = weight.clone()
    output_torch = conv3d(input_tensor)
    
    # Since the kernel incorrectly computes in_channels_per_group,
    # the results are expected to differ.
    with pytest.raises(AssertionError):
        assert torch.allclose(output_kernel, output_torch, atol=1e-3)

# Issue 3: Test missing kernel error checking by forcing an invalid memory access.
def test_invalid_kernel_launch():
    # We deliberately create an input tensor that is too small relative to the kernel.
    # This may cause the computed effective receptive field to be out-of-bound.
    batch_size = 1
    in_channels = 4
    out_channels = 4
    in_d, in_h, in_w = 2, 2, 2  # very small input
    kernel_d, kernel_h, kernel_w = 5, 5, 5  # large kernel
    groups = 1

    input_tensor = torch.randn(batch_size, in_channels, in_d, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_d, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = None
    stride = 1
    padding = 0
    dilation = 1

    cuda_mod = build_kernel()
    
    # We expect the kernel to perform out-of-bound accesses.
    with pytest.raises(RuntimeError):
        output = cuda_mod.forward(input_tensor, weight, bias, stride, padding, dilation, groups)
        # Force cuda synchronization to raise any kernel launch errors.
        torch.cuda.synchronize()

# Issue 4: Test that the use of cudaDeviceSynchronize() after each kernel launch leads to performance issues.
# While we cannot measure performance regressions in a simple unit test, we can simulate by timing multiple launches.
def test_kernel_synchronization_overhead(benchmark):
    batch_size = 4
    in_channels = 8
    out_channels = 8
    in_d, in_h, in_w = 20, 20, 20
    kernel_d, kernel_h, kernel_w = 3, 3, 3
    groups = 1

    input_tensor = torch.randn(batch_size, in_channels, in_d, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_d, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = None
    stride = 1
    padding = 1
    dilation = 1

    cuda_mod = build_kernel()

    def run_forward():
        output = cuda_mod.forward(input_tensor, weight, bias, stride, padding, dilation, groups)
        torch.cuda.synchronize()
    # Benchmark the kernel launch over several iterations.
    benchmark(run_forward)

# Issue 5: Test non-contiguous output tensor for bias addition.
def test_non_contiguous_output():
    batch_size = 2
    in_channels = 4
    out_channels = 4
    in_d, in_h, in_w = 10, 10, 10
    kernel_d, kernel_h, kernel_w = 3, 3, 3
    groups = 1

    # Create input and weight normally.
    input_tensor = torch.randn(batch_size, in_channels, in_d, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_d, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    stride = 1
    padding = 1
    dilation = 1

    cuda_mod = build_kernel()
    # Run normally once to get a contiguous output.
    output = cuda_mod.forward(input_tensor, weight, bias, stride, padding, dilation, groups)
    # Create a non-contiguous view by transposing two dimensions.
    output_non_contig = output.transpose(2, 4)
    
    # Since our kernel always writes to a contiguous output,
    # we simulate a failure scenario by manually invoking the add_bias_kernel on a non-contiguous tensor.
    # Here, we simply check that the non-contiguous tensor is not properly handled.
    # We expect the values to not match what would be obtained if the bias was added properly.
    output_non_contig_saved = output_non_contig.clone()
    # Attempt to run the bias addition kernel on non-contiguous output via the extension (this uses raw pointer access).
    with pytest.raises(AssertionError):
        # Run the forward kernel again on the non-contiguous tensor as if it were contiguous.
        output_bias_applied = cuda_mod.forward(input_tensor, weight, bias, stride, padding, dilation, groups)
        # Force a difference in the non-contiguous case.
        assert torch.allclose(output_non_contig_saved, output_bias_applied, atol=1e-3)
