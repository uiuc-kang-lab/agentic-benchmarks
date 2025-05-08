
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to compile and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case to trigger Issue 1: Incorrect handling of output_padding.
# We set output_padding to a nonzero value so that the native PyTorch conv_transpose2d
# produces a result that our kernel (which ignores output_padding) will not match.
def test_output_padding_mismatch():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 6
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1  # non-zero output_padding to trigger the issue
    
    # Input tensor
    x = torch.randn(batch_size, in_channels, 8, 8, device='cuda')
    
    # Create weight and bias tensors with the shape expected by ConvTranspose2d:
    # weight shape: [in_channels, out_channels/groups, kernel_size, kernel_size].
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device='cuda')
    bias = torch.randn(out_channels, device='cuda')
    
    # Build the CUDA kernel output.
    out_kernel = cuda_module.forward(
        x, weight, bias, 
        stride=stride, padding=padding, output_padding=output_padding, groups=1
    )
    
    # Build the reference output using PyTorch's native ConvTranspose2d.
    ref_conv = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=stride, 
        padding=padding, output_padding=output_padding, bias=True
    ).cuda()
    # Manually set the weight and bias so that they match our test inputs.
    ref_conv.weight.data = weight
    ref_conv.bias.data = bias
    out_ref = ref_conv(x)
    
    # The two outputs are expected to differ because our kernel does not adjust for output_padding.
    # We deliberately assert that they should not be close.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), "Kernel unexpectedly handled output_padding correctly!"

# Test case to trigger Issue 2: Kernel only supports float32 while input is provided as double.
def test_input_tensor_type():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 6
    kernel_size = 3
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    # Create double precision input, weight, bias.
    x = torch.randn(batch_size, in_channels, 10, 10, device='cuda', dtype=torch.double)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device='cuda', dtype=torch.double)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.double)
    
    with pytest.raises(RuntimeError):
        # The call should raise an error because the kernel calls data_ptr<float>()
        cuda_module.forward(x, weight, bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups)

# Test case to trigger Issue 3: Lack of validation for groups division.
# We deliberately set groups such that in_channels is not divisible by groups.
def test_invalid_groups():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 5   # intentionally choosing a number NOT divisible by groups
    out_channels = 6
    kernel_size = 3
    stride = 1
    padding = 0
    output_padding = 0
    groups = 2       # 5 % 2 != 0
    
    # Create input, weight, bias tensors.
    x = torch.randn(batch_size, in_channels, 10, 10, device='cuda', dtype=torch.float32)
    # For ConvTranspose2d in PyTorch, weight shape is [in_channels, out_channels/groups, kernel_size, kernel_size].
    # Here, since in_channels is not divisible by groups, the proper behavior would be to raise an error.
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    
    # Our kernel does not do a proper sanity check on groups, so it may either produce an incorrect result
    # or silently lead to out-of-bound memory accesses. Here we compare with the PyTorch ConvTranspose2d,
    # which will throw an exception.
    ref_exception = None
    try:
        ref_conv = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=True
        ).cuda()
        # This is likely to raise an error because in_channels is not divisible by groups.
        ref_conv.weight.data = weight
        ref_conv.bias.data = bias
        out_ref = ref_conv(x)
    except Exception as e:
        ref_exception = e

    # Now call the CUDA kernel forward.
    # Depending on how the kernel behaves, it may not raise an exception even though the math is invalid.
    out_kernel = cuda_module.forward(x, weight, bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups)
    
    # If PyTorch raised an error, we expect our kernel to not match a proper output.
    # We check that our kernel’s output is different from a fallback output shape (which may be zero or garbage).
    # One way to do this is to ensure that the kernel output is not close to any valid reference.
    # Here, for simplicity, we assert that if PyTorch failed, the kernel result should not be close to zeros.
    # (This test may pass if the kernel silently computes something, but the correctness is compromised.)
    assert not torch.allclose(out_kernel, torch.zeros_like(out_kernel)), \
        "Kernel produced an output that looks valid even with invalid groups configuration!"

if __name__ == "__main__":
    pytest.main([__file__])
