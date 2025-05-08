
import torch
import pytest
from torch.utils.cpp_extension import load

# A helper to build the kernel module from kernel.cu.
def build_kernel():
    module = load(
        name="conv_transpose2d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: The kernel only supports float inputs.
def test_non_float_input():
    module = build_kernel()
    # Create double precision input and weight tensors.
    batch_size = 2
    in_channels = 3
    out_channels = 4
    inH, inW = 8, 8
    kernelH, kernelW = 3, 3
    stride = [1, 1]
    padding = [1, 1]
    
    # Create tensors in double precision.
    input_tensor = torch.randn(batch_size, in_channels, inH, inW, dtype=torch.double, device="cuda")
    weight_tensor = torch.randn(in_channels, out_channels, kernelH, kernelW, dtype=torch.double, device="cuda")
    bias_tensor = torch.randn(out_channels, dtype=torch.double, device="cuda")
    
    with pytest.raises(RuntimeError):
        # Expect an error because the kernel code uses float and does not check dtype.
        out = module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding)
        torch.cuda.synchronize()

# Issue 2: The kernel expects contiguous tensors.
def test_non_contiguous_inputs():
    module = build_kernel()
    batch_size = 2
    in_channels = 3
    out_channels = 4
    inH, inW = 8, 8
    kernelH, kernelW = 3, 3
    stride = [1, 1]
    padding = [1, 1]

    # Create contiguous tensors first.
    input_tensor = torch.randn(batch_size, in_channels, inH, inW, device="cuda", dtype=torch.float32)
    weight_tensor = torch.randn(in_channels, out_channels, kernelH, kernelW, device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # Make the tensors non-contiguous by transposing a couple of dimensions.
    input_non_contig = input_tensor.transpose(1, 2)
    weight_non_contig = weight_tensor.transpose(1, 2)
    
    # Run the custom kernel and compare with PyTorch's own ConvTranspose2d.
    conv_transpose = torch.nn.ConvTranspose2d(in_channels, out_channels, (kernelH, kernelW),
                                                stride=stride, padding=padding, bias=True).cuda()
    # Force the conv_transpose2d weights to match our weight_non_contig.
    with torch.no_grad():
        conv_transpose.weight.copy_(weight_tensor)
        conv_transpose.bias.copy_(bias_tensor)
    # Reference output
    ref_output = conv_transpose(input_tensor)
    
    # The custom CUDA kernel is not designed to handle non-contiguous inputs.
    # We thus expect the result to be different from the reference.
    custom_output = module.forward(input_non_contig, weight_non_contig, bias_tensor, stride, padding)
    torch.cuda.synchronize()
    # Here, instead of using allclose, we expect a significant difference.
    assert not torch.allclose(custom_output, ref_output, atol=1e-4), \
        "Kernel unexpectedly produced the same result even with non-contiguous inputs."

# Issue 3: No stride validation causes division-by-zero when stride is zero.
def test_zero_stride_error():
    module = build_kernel()
    batch_size = 2
    in_channels = 3
    out_channels = 4
    inH, inW = 8, 8
    kernelH, kernelW = 3, 3
    # Use zero stride which will cause division-by-zero inside the kernel.
    stride = [0, 0]
    padding = [1, 1]

    input_tensor = torch.randn(batch_size, in_channels, inH, inW, device="cuda", dtype=torch.float32)
    weight_tensor = torch.randn(in_channels, out_channels, kernelH, kernelW, device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    with pytest.raises(Exception):
        _ = module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding)
        torch.cuda.synchronize()

# Issue 4: Loop unrolling on non-constant loop bounds.
# Although this does not cause an outright error, it might lead to performance degradation.
# This test compares the custom kernel output with PyTorch's output using a runtime‐determined kernel size.
def test_runtime_kernel_size():
    module = build_kernel()
    batch_size = 2
    in_channels = 3
    out_channels = 4
    inH, inW = 16, 16
    # Use a non-standard (runtime) kernel size.
    kernelH, kernelW = 4, 7
    stride = [1, 1]
    padding = [1, 2]

    input_tensor = torch.randn(batch_size, in_channels, inH, inW, device="cuda", dtype=torch.float32)
    weight_tensor = torch.randn(in_channels, out_channels, kernelH, kernelW, device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Use PyTorch's functional conv_transpose2d as reference.
    ref_output = torch.nn.functional.conv_transpose2d(input_tensor, weight_tensor, bias=bias_tensor,
                                                      stride=stride, padding=padding)
    custom_output = module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding)
    torch.cuda.synchronize()

    # Even though correctness is maintained, the use of #pragma unroll with runtime bounds
    # might impact performance. Here we just check correctness.
    assert torch.allclose(custom_output, ref_output, atol=1e-4), \
        "Custom kernel output differs from PyTorch output for runtime-determined kernel sizes."
