
import torch
import pytest
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

# Issue 1: Kernel only supports float32 -- using double input should trigger an error or wrong result.
def test_dtype_issue():
    mod = build_kernel()
    # Create a double-precision input tensor
    x = torch.randn(1, 3, 10, 10, device="cuda", dtype=torch.float64)
    # Weight shape: [in_channels, out_channels, kernel_size, kernel_size] when groups=1.
    weight = torch.randn(3, 6, 3, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(6, device="cuda", dtype=torch.float32)
    with pytest.raises(Exception):
        # Kernel expects float32 input; passing double may cause memory corruption or unexpected behavior.
        out = mod.forward(x, weight, bias, 1, 0, 0, 1)
        torch.cuda.synchronize()

# Issue 2: Kernel does not verify device placement -- passing a CPU tensor should cause an error.
def test_device_issue():
    mod = build_kernel()
    x = torch.randn(1, 3, 10, 10, device="cpu", dtype=torch.float32)  # CPU tensor
    weight = torch.randn(3, 6, 3, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(6, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        out = mod.forward(x, weight, bias, 1, 0, 0, 1)

# Issue 3: in_channels not divisible by groups -- results will be wrong due to integer division in the kernel.
def test_group_divisibility_issue():
    mod = build_kernel()
    # in_channels is not divisible by groups. For example, if in_channels=5 and groups=2, then 5//2 = 2, so one channel is dropped.
    batch_size = 1
    in_channels = 5
    out_channels = 6  # For groups=2, weight shape should be [in_channels, out_channels//groups, ...]
    x = torch.randn(batch_size, in_channels, 10, 10, device="cuda", dtype=torch.float32)
    # Even though PyTorch's ConvTranspose2d would check the channel grouping,
    # our kernel does not so we build weight with shape (5, 3, 3, 3)
    weight = torch.randn(in_channels, out_channels // 2, 3, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    out_kernel = mod.forward(x, weight, bias, 1, 0, 0, 2)
    # Build a reference module from PyTorch
    conv = torch.nn.ConvTranspose2d(in_channels, out_channels, 3, stride=1, padding=0,
                                    output_padding=0, groups=2, bias=True).to("cuda")
    # Manually set conv weights to match (note: ordering differences may cause a mismatch)
    with pytest.raises(AssertionError):
        out_ref = conv(x)
        # The computed output is expected to differ from the reference because one channel is effectively dropped.
        torch.testing.assert_allclose(out_kernel, out_ref)

# Issue 4: No error-checking after kernel launch -- if the grid dimension exceeds the device limit, a launch failure occurs.
def test_grid_dimension_issue():
    mod = build_kernel()
    # Many CUDA devices limit gridDim.z to 65535.
    # Force gridDim.z = batch_size * out_channels to exceed that limit.
    batch_size = 100
    out_channels = 700   # 100 * 700 = 70000 > 65535
    in_channels = 4
    h_in, w_in = 8, 8
    x = torch.randn(batch_size, in_channels, h_in, w_in, device="cuda", dtype=torch.float32)
    # Weight shape: [in_channels, out_channels, kernel_size, kernel_size] (groups=1)
    weight = torch.randn(in_channels, out_channels, 3, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        out = mod.forward(x, weight, bias, 1, 0, 0, 1)
        torch.cuda.synchronize()
