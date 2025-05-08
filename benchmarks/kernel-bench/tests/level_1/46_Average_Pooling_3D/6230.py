
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="stream_optimized_avgpool3d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Kernel only supports float32; test that using float16 input yields incorrect results.
def test_input_dtype_support():
    # Create a tensor in float16, which the kernel does not support.
    batch_size, channels, depth, height, width = 2, 3, 10, 10, 10
    x = torch.randn(batch_size, channels, depth, height, width, device="cuda", dtype=torch.float16)
    # Use valid pooling parameters.
    kernel_size = 3
    stride = 2
    padding = 1

    # Built-in PyTorch layer for comparison.
    ref_pool = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    ref_output = ref_pool(x)

    mod = build_kernel()
    # Our CUDA kernel always assumes float32.
    out = mod.forward(x, kernel_size, stride, padding)
    torch.cuda.synchronize()
    # The outputs should differ since the kernel misinterprets the data.
    with pytest.raises(AssertionError):
        assert torch.allclose(out, ref_output, atol=1e-3), "Kernel did not mis-handle non-float32 input as expected."

# Issue 2: Kernel only accepts cubic pooling parameters. Test that passing non-integer (tuple) pooling parameters fails.
def test_non_cubic_pooling_parameters():
    batch_size, channels, depth, height, width = 2, 3, 12, 16, 20
    x = torch.randn(batch_size, channels, depth, height, width, device="cuda", dtype=torch.float32)
    # Here, instead of an int, we try to pass tuples to simulate non-cubic situations.
    kernel_size = (3, 4, 5)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    mod = build_kernel()

    # Since our binding expects ints, we simulate the error by manually checking the type.
    with pytest.raises(TypeError):
        mod.forward(x, kernel_size, stride, padding)

# Issue 3: Manual stream management. Test that invoking the kernel does not take advantage of the current stream.
def test_stream_management():
    # Create an input and record the current stream.
    batch_size, channels, depth, height, width = 2, 3, 20, 20, 20
    x = torch.randn(batch_size, channels, depth, height, width, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 1
    padding = 1

    current_stream = torch.cuda.current_stream(device="cuda")
    mod = build_kernel()
    out = mod.forward(x, kernel_size, stride, padding)
    torch.cuda.synchronize()
    # Check that the output shape is as expected
    expected_depth = (depth + 2 * padding - kernel_size) // stride + 1
    expected_height = (height + 2 * padding - kernel_size) // stride + 1
    expected_width = (width + 2 * padding - kernel_size) // stride + 1
    assert out.shape == (batch_size, channels, expected_depth, expected_height, expected_width), \
        f"Output shape {out.shape} unexpected."

    # Now, check that the current stream has not been altered by our kernel.
    # (Because the kernel created its own stream instead of using the current one,
    #  the current stream should remain the same.)
    assert torch.cuda.current_stream(device="cuda").cuda_stream == current_stream.cuda_stream, \
        "Current CUDA stream was modified by the kernel."

# Issue 4: Border handling. Test that the branchless clamping leads to a (possibly) incorrect average at the borders.
def test_border_behavior():
    # Create an input tensor where border elements are distinct.
    batch_size, channels, depth, height, width = 1, 1, 5, 5, 5
    x = torch.arange(depth * height * width, dtype=torch.float32, device="cuda").reshape(1, 1, depth, height, width)
    kernel_size = 3
    stride = 1
    padding = 1

    # Using PyTorch's built-in AvgPool3d (which uses count_include_pad=True by default)
    ref_pool = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    ref_output = ref_pool(x)

    mod = build_kernel()
    out = mod.forward(x, kernel_size, stride, padding)
    torch.cuda.synchronize()
    # We expect a small numerical difference if the border handling were correct.
    # If the border elements are mis‐averaged, then the maximum absolute difference will be nonzero.
    diff = (out - ref_output).abs().max()
    assert diff > 1e-3, f"Border behavior appears correct (diff={diff}); expected a discrepancy due to branchless clamping."

