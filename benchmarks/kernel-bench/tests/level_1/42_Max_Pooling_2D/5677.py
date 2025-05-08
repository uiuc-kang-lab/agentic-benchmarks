
import pytest
import torch
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

# Issue 1: Test to trigger the wrong function overload due to fmaxf on double tensors.
def test_double_precision_input():
    # Create a double tensor input suited for pooling.
    batch_size = 2
    channels = 3
    height = 10
    width = 10
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, channels, height, width, dtype=torch.double, device="cuda")
    module = build_kernel()
    
    # The kernel is launched via this module.forward interface, which expects float types.
    # For double, using fmaxf can be problematic.
    with pytest.raises(RuntimeError):
        # We expect a runtime failure or misbehavior when using double.
        _ = module.forward(x, kernel_size, stride, padding, dilation)

# Issue 2: Test to trigger grid dimension problems when batch_size*channels exceeds gridDim.z limit.
def test_large_grid_dimension():
    # We choose tensor dimensions such that batch_size*channels > 65535, which is commonly the limit.
    # To avoid huge memory usage, we pick minimal spatial size.
    batch_size = 70000  # deliberately large to force gridDim.z > 65535 with channels=1.
    channels = 1
    height = 3
    width = 3
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1

    # Create a tensor with minimal spatial dims.
    x = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device="cuda")
    module = build_kernel()
    
    # The following should raise an error during kernel launch due to grid dimension limits.
    with pytest.raises(RuntimeError):
        _ = module.forward(x, kernel_size, stride, padding, dilation)
        
if __name__ == '__main__':
    pytest.main([__file__])
