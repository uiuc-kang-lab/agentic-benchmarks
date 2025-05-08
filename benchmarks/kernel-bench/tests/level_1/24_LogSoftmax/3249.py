
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Trigger potential issue with unqualified "max" by using extreme values.
def test_extreme_values():
    # Create an input where the maximum should be very obvious.
    # Using extreme negative values on the left and a high value on the right.
    batch_size, dim = 4, 1024
    a = torch.full((batch_size, dim), -1e30, device='cuda', dtype=torch.float32)
    # Set one element per row to a high value.
    for i in range(batch_size):
        a[i, i] = 1e30
    my_module = build_kernel()
    out = my_module.forward(a, 1)
    # Expected output matches torch.log_softmax.
    expected = torch.log_softmax(a, dim=1)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, atol=1e-4), "Extreme values: kernel output does not match expected torch.log_softmax."

# Test case 2: Trigger issue with use of std::numeric_limits<>::infinity()
# by using inputs with very negative numbers.
def test_negative_infinities():
    # Create a tensor where all values are very low
    batch_size, dim = 3, 512
    a = torch.full((batch_size, dim), -1e20, device='cuda', dtype=torch.float32)
    # But one value per row is set to 0.
    for i in range(batch_size):
        a[i, -1] = 0.
    my_module = build_kernel()
    out = my_module.forward(a, 1)
    expected = torch.log_softmax(a, dim=1)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, atol=1e-4), "Negative infinity: kernel output does not handle infinities as expected."

# Test case 3: Trigger issue with NaN propagation.
def test_nan_input():
    batch_size, dim = 2, 256
    a = torch.randn(batch_size, dim, device='cuda', dtype=torch.float32)
    # Inject a NaN value in every row.
    a[0, 10] = float('nan')
    a[1, 20] = float('nan')
    my_module = build_kernel()
    out = my_module.forward(a, 1)
    # In torch.log_softmax, any row containing NaN will lead to NaN in every element.
    expected = torch.log_softmax(a, dim=1)
    torch.cuda.synchronize()
    # Check that the positions that should be nan are nan.
    assert torch.isnan(out[0, 0]) and torch.isnan(out[1, 0]), "NaN input: kernel did not produce NaNs as expected in the output."

# Test case 4: Test double precision input for atomicAdd support.
def test_double_precision():
    batch_size, dim = 4, 1024
    # Only run this test if the device supports double-precision atomicAdd.
    cc = torch.cuda.get_device_capability()
    if cc[0] < 6:
        pytest.skip("Double precision atomicAdd not supported on devices with compute capability < 6.0")
    a = torch.randn(batch_size, dim, device='cuda', dtype=torch.float64)
    my_module = build_kernel()
    out = my_module.forward(a, 1)
    expected = torch.log_softmax(a, dim=1)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, atol=1e-8), "Double precision: kernel output does not match expected torch.log_softmax."

if __name__ == "__main__":
    pytest.main([__file__])
