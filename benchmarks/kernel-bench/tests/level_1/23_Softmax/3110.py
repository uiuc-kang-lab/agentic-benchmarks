
import pytest
import torch
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

# Issue 1: Uninitialized shared memory when num_features < THREADS_PER_BLOCK.
def test_uninitialized_shared_memory():
    # Use a small number of features to force some threads to not participate in the loop.
    batch_size = 1
    num_features = 10  # THREADS_PER_BLOCK is defined as 256 in the kernel.
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float32)
    # Expected output computed by torch.softmax
    expected = torch.softmax(x, dim=1)
    
    kernel_module = build_kernel()
    y = kernel_module.forward(x)
    torch.cuda.synchronize()
    
    # The result may be incorrect due to uninitialized shared memory access.
    assert not torch.allclose(y, expected, atol=1e-4), \
        "Test expected a mismatch due to uninitialized shared memory, but the results matched."

# Issue 2: Kernel only supports 2D tensors along dim1.
def test_tensor_dimensionality():
    # Provide an input tensor that is 3D, which should trigger the TORCH_CHECK in forward.
    x = torch.randn(2, 3, 4, device="cuda", dtype=torch.float32)
    kernel_module = build_kernel()
    
    with pytest.raises(RuntimeError, match="Input tensor must be 2D."):
        _ = kernel_module.forward(x)

# Issue 3: Kernel expects float32; providing another dtype should trigger a check.
def test_tensor_dtype():
    # Provide an input tensor in float64 which should trigger the TORCH_CHECK in forward.
    x = torch.randn(2, 10, device="cuda", dtype=torch.float64)
    kernel_module = build_kernel()
    
    with pytest.raises(RuntimeError, match="Input tensor must be float32."):
        _ = kernel_module.forward(x)
