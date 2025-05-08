
import torch
import pytest
from torch.nn import AvgPool3d
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension module from kernel.cu
    return load(
        name="avg_pool3d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )

@pytest.fixture(scope="module")
def cuda_module():
    mod = build_kernel()
    return mod

@pytest.fixture
def default_params():
    # Use typical parameters for a 3D pooling: kernel_size=3, stride=2, padding=1
    return 3, 2, 1

def test_shared_memory_uninitialized(cuda_module, default_params):
    # Issue 1 & 3: When the pooling window is on a border, the kernel may use uninitialized shared memory.
    batch_size = 2
    channels = 3
    depth = 5   # deliberately small so that padding (or border) plays a role
    height = 8
    width = 8
    kernel_size, stride, padding = default_params

    # Create an input where pooling windows at the border will be partially outside.
    input_tensor = torch.randn(batch_size, channels, depth, height, width, device='cuda', dtype=torch.float32)
    
    # Run our custom CUDA kernel
    output_kernel = cuda_module.forward(input_tensor, kernel_size, stride, padding)
    
    # Create a reference using PyTorch's implementation.
    avg_pool = AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    output_ref = avg_pool(input_tensor)
    
    # We expect difference due to uninitialized shared memory in border pooling windows.
    # The test is designed to fail if the kernel were correct.
    # Here we check that the custom kernel DOES NOT match the CPU output.
    # (In a fixed implementation the two outputs would be nearly equal.)
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), (
        f"Expected mismatch due to uninitialized shared memory usage, but outputs matched."
    )

def test_input_tensor_type(cuda_module, default_params):
    # Issue 2: Kernel only supports float32. Passing a different type should trigger an error.
    batch_size = 2
    channels = 3
    depth = 8
    height = 8
    width = 8
    kernel_size, stride, padding = default_params

    # Create an input tensor of type float64 (double) which is not supported.
    input_tensor = torch.randn(batch_size, channels, depth, height, width, device='cuda', dtype=torch.float64)
    
    with pytest.raises(RuntimeError):
        # Expect an error because the kernel is written to work only on float32.
        cuda_module.forward(input_tensor, kernel_size, stride, padding)
