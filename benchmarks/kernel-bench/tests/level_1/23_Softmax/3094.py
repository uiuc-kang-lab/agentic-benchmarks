
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="softmax_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

def call_forward(x: torch.Tensor):
    # Loads the extension and calls the forward function.
    kernel_module = build_kernel()
    return kernel_module.forward(x)

def test_cuda_launch_error_checking():
    # Test case to trigger an error by using an input with zero features.
    # The kernel does not handle num_features == 0 and division by zero may occur.
    batch_size = 4
    num_features = 0  # Zero features edge-case; expected to trigger abnormal behavior.
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError):
        # The kernel should ideally fail or raise an error.
        y = call_forward(x)
        # Force CUDA synchronization to capture any async errors.
        torch.cuda.synchronize()

def test_input_tensor_type():
    # Test case to ensure the kernel rejects inputs of wrong type (not float32)
    batch_size = 4
    num_features = 128
    # Create input in double precision to trigger the TORCH_CHECK failure.
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float64)
    
    with pytest.raises(RuntimeError) as exc_info:
        y = call_forward(x)
    # Optionally check the error message for expected text.
    assert "float32" in str(exc_info.value)
