
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper function to build the CUDA extension module
def build_kernel():
    cwd = os.path.dirname(os.path.abspath(__file__))
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(cwd, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# 1. Test Issue: Horizontal kernel dimension ignored
#    We create a weight tensor with a horizontal extent > 1.
#    The kernel, however, only reads from the vertical slice.
#    In a correct implementation, the output would incorporate both kernel dimensions.
def test_horizontal_kernel_ignored():
    cuda_module = build_kernel()
    # Setup a scenario with asymmetric kernel: height > 1 and width = 2
    batch = 1
    channels = 1
    in_h = 10
    in_w = 10
    stride = 1
    padding = 0
    dilation = 1
    # Create an input filled with ones for simplicity.
    x = torch.ones(batch, channels, in_h, in_w, device="cuda", dtype=torch.float32)
    
    # Define a weight tensor intended for a 2D kernel.
    # Even though the PyTorch forward used in Conv2d expects a 4D weight, our kernel only uses kernel.size(2).
    # We simulate the scenario by creating a 4D weight with kernel width > 1.
    kernel_h = 3
    kernel_w = 2
    # In a proper 2D convolution, output would be the sum over all 6 elements.
    # However, our kernel only loads kernel_h (3) values and ignores the horizontal component.
    # We set the vertical weights to 1 and horizontal weights to a distinguishable constant (e.g. 10).
    weight = torch.ones(channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    # Bias: zero for simplicity.
    bias = torch.zeros(channels, device="cuda", dtype=torch.float32)

    # Call the kernel forward.
    out = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups=channels)
    
    # Expected output: In a full 2D convolution, each output element would be sum over a 3x2 region (i.e. 6).
    # Our kernel only sums 3 elements (the vertical ones, each multiplied by 1).
    # Thus, we expect a discrepancy.
    expected_full_conv = 6.0  # correct sum for a full convolution window of ones.
    expected_kernel_impl = 3.0  # sum computed by the implemented kernel.
    
    # Check that the output does not match the correct full convolution sum.
    assert not torch.allclose(out, torch.full_like(out, expected_full_conv)), (
        "Kernel erroneously handled horizontal kernel dimension although it should ignore it."
    )
    # Also check that it matches the incorrect expected value.
    assert torch.allclose(out, torch.full_like(out, expected_kernel_impl)), (
        "Kernel output does not match the expected (incomplete) computation."
    )

# 2. Test Issue: Incorrect computation of output width dimension.
#    We purposely choose input dimensions and padding such that the correct output width differs from what the kernel computes.
def test_wrong_out_w_calculation():
    cuda_module = build_kernel()
    batch = 1
    channels = 1
    in_h = 10
    in_w = 12  # pick a width that would be sensitive to proper calculation
    stride = 1
    # Use padding and dilation such that the correct formula for width would be:
    # out_w_correct = (in_w + 2*padding - dilation*(kernel_w-1) - 1)//stride + 1.
    # We simulate a kernel with horizontal size 2 (even though our kernel ignores horizontal).
    padding = 1
    dilation = 2
    kernel_h = 3
    kernel_w = 2
    x = torch.ones(batch, channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.ones(channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.zeros(channels, device="cuda", dtype=torch.float32)
    
    out = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups=channels)
    
    # Correct expected out_w if the horizontal kernel component were computed properly:
    out_w_correct = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    # But the kernel computes:
    out_w_in_kernel = (in_w + 2 * padding - 1) // stride + 1
    
    # In our test case, these two values should differ.
    assert out_w_correct != out_w_in_kernel, "Test configuration does not produce differing output width!"
    # The kernel's output shape will have out_w_in_kernel columns.
    output_shape = out.shape  # (batch, channels, out_h, out_w)
    
    # Check that the computed out_w in output shape matches the kernel's miscalculation.
    assert output_shape[3] == out_w_in_kernel, "Kernel output width does not match its miscalculation."
    
    # Also check that it does not equal the correct output width.
    assert output_shape[3] != out_w_correct, "Kernel accidentally computed correct output width."

# 3. Test Issue: Shared memory weight loading does not handle kernel_h larger than blockDim.x * blockDim.y.
#    We simulate this by creating a kernel with very large kernel_h.
def test_excessive_kernel_h():
    cuda_module = build_kernel()
    batch = 1
    channels = 1
    in_h = 50
    in_w = 50
    stride = 1
    padding = 2
    dilation = 1
    # Choose a very tall kernel_h that exceeds the thread block capacity (tile_x=32, tile_y=8 => 256 threads)
    kernel_h = 300  
    kernel_w = 1  # keep horizontal dim 1 so that we only worry about vertical kernel dimension
    x = torch.ones(batch, channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.ones(channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.zeros(channels, device="cuda", dtype=torch.float32)

    # The implemented kernel loads weights with a condition "if (tid < kernel_h)".
    # In this case, only the first 256 threads (if blockDim.x*blockDim.y==256) will load a weight,
    # leaving the remaining kernel weights uninitialized (or not loaded),
    # which should cause the output to be incorrect (likely lower than expected).
    out = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups=channels)
    
    # For comparison, compute what would happen if all kernel weights were applied.
    # Since the entire input region is ones and weight values are ones,
    # the correct convolution sum (ignoring boundaries) would be kernel_h.
    # However, the kernel likely only sums 256 elements.
    expected_full_sum = float(kernel_h)
    expected_partial_sum = 256.0  # assuming tile size 256 handles the first 256 weights
    # We check that the output is closer to the partial sum.
    # We only test one element (at a central position) to reduce border effects.
    central_val = out[0, 0, 2, 2].item()
    assert abs(central_val - expected_partial_sum) < 1e-3, (
        f"Kernel did not load weights correctly for kernel_h > block threads. "
        f"Expected about {expected_partial_sum}, got {central_val}"
    )

# 4. Test Issue: Kernel only supports float32 input.
#    Provide double precision (float64) input to trigger a failure due to type mismatch.
def test_input_tensor_wrong_dtype():
    cuda_module = build_kernel()
    batch = 1
    channels = 1
    in_h = 10
    in_w = 10
    stride = 1
    padding = 0
    dilation = 1
    kernel_h = 3
    kernel_w = 1
    # Create input tensor with double dtype.
    x = torch.ones(batch, channels, in_h, in_w, device="cuda", dtype=torch.float64)
    weight = torch.ones(channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float64)
    bias = torch.zeros(channels, device="cuda", dtype=torch.float64)
    
    with pytest.raises(RuntimeError):
        # This call should fail because the kernel uses data_ptr<float>()
        cuda_module.forward(x, weight, bias, stride, padding, dilation, groups=channels)
