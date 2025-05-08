
import pytest
import torch
from torch.utils.cpp_extension import load

# This helper builds the CUDA extension from kernel.cu.
def build_kernel():
    return load(
        name="custom_maxpool1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# Issue 1: Kernel only supports float32.
# Test: Pass a double-precision (float64) tensor and expect the kernel to produce wrong values or error.
def test_input_dtype():
    my_module = build_kernel()
    batch_size, channels, length = 8, 4, 16
    # create a double tensor on CUDA that is contiguous.
    x = torch.randn(batch_size, channels, length, dtype=torch.float64, device='cuda')
    # Kernel expects float32; this misuse may result in a CUDA error or wrong results.
    with pytest.raises(RuntimeError):
        # We do not expect the kernel to work for float64.
        my_module.forward(x, 3, 1, 0, 1, False)

# Issue 2: Incorrect output format when return_indices is true.
# Test: Compare the output shape of the custom kernel with the expected output from nn.MaxPool1d.
def test_return_indices_format():
    my_module = build_kernel()
    batch_size, channels, length = 8, 4, 32
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = True
    x = torch.randn(batch_size, channels, length, dtype=torch.float32, device='cuda')
    
    # Compute expected output using the native PyTorch MaxPool1d (which returns a tuple when return_indices=True)
    maxpool = torch.nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices=True).to(x.device)
    expected_output, expected_indices = maxpool(x)
    
    # Now call the custom kernel.
    out = my_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    # The kernel returns a single tensor that is a concatenation of output and indices along the last dimension.
    # Thus its last dimension will be twice as long.
    assert out.shape[-1] == expected_output.shape[-1] * 2, (
        f"Expected last dimension to be {expected_output.shape[-1]*2} but got {out.shape[-1]}"
    )

# Issue 3: Kernel launch fails for large batch sizes due to use of blockIdx.z.
# Test: Create an input with a batch size exceeding typical gridDim.z limits.
def test_large_batch():
    my_module = build_kernel()
    # Typical CUDA gridDim.z limit is 65535. We use a batch size larger than that.
    batch_size = 70000
    channels = 2
    length = 32
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False
    x = torch.randn(batch_size, channels, length, dtype=torch.float32, device='cuda')
    
    with pytest.raises(RuntimeError):
        # This should trigger a kernel launch failure because gridDim.z (batch index) exceeds the limit.
        my_module.forward(x, kernel_size, stride, padding, dilation, return_indices)

# Issue 4: No error checking after kernel launch.
# Test: Pass a noncontiguous input tensor (which the host function checks) to trigger an error before launching the kernel.
def test_non_contiguous_input():
    my_module = build_kernel()
    batch_size, channels, length = 8, 4, 32
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False
    x = torch.randn(batch_size, channels, length, dtype=torch.float32, device='cuda')
    # Make the tensor noncontiguous.
    x = x.transpose(1, 2)
    with pytest.raises(RuntimeError, match="Input must be contiguous"):
        my_module.forward(x, kernel_size, stride, padding, dilation, return_indices)

# Issue 5: Branch divergence due to dynamic check of return_indices inside the kernel.
# Test: Run the kernel with both return_indices True and False on the same input and verify that at least the output “values” are identical.
def test_return_value_consistency():
    my_module = build_kernel()
    batch_size, channels, length = 8, 4, 32
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    x = torch.randn(batch_size, channels, length, dtype=torch.float32, device='cuda')
    
    # Get output with return_indices False.
    out_no_idx = my_module.forward(x, kernel_size, stride, padding, dilation, False)
    
    # Get output with return_indices True.
    out_with_idx = my_module.forward(x, kernel_size, stride, padding, dilation, True)
    # Extract the first half of the last dimension which corresponds to pooled values.
    pooled_with_idx = out_with_idx[..., :out_no_idx.shape[-1]]
    
    # Even though there is an extra branch, the actual pooling result (values) should be exactly the same.
    assert torch.allclose(out_no_idx, pooled_with_idx, atol=1e-5), "Pooled values differ between modes!"

# Issue 6: Fixed thread block configuration may be suboptimal for inputs with nonstandard shapes.
# Test: Use an input whose output dimensions are not multiples of the chosen thread block size and compare against PyTorch's native MaxPool1d.
def test_non_standard_output_shape():
    my_module = build_kernel()
    batch_size, channels, length = 10, 7, 45  # dimensions chosen to likely produce non-multiple of 32 or 4
    kernel_size = 4
    stride = 3
    padding = 1
    dilation = 2
    return_indices = False
    x = torch.randn(batch_size, channels, length, dtype=torch.float32, device='cuda')
    
    # Compute expected output using the native PyTorch MaxPool1d.
    maxpool = torch.nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices=False).to(x.device)
    expected_output = maxpool(x)
    
    # Get output from the custom kernel.
    out = my_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    
    # There might be numerical differences if the thread block configuration is not flexible.
    # We check if the maximum absolute difference exceeds a tolerance.
    diff = (out - expected_output).abs().max().item()
    assert diff < 1e-4, f"Output differs from expected! Max difference = {diff}"
