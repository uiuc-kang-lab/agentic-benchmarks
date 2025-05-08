
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # This will compile and load the CUDA extension from the file kernel.cu
    cuda_module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_weight_size_exceeds_constant_memory():
    # Issue 1: Test that the kernel enforces the constant memory weight size limit.
    # Construct a weight tensor with more than 1024 elements.
    # For example: in_channels = 8, group_size_out = 4, kernel_size = 40 => total elements = 8*4*40 = 1280 > 1024.
    in_channels = 8
    group_size_out = 4  # weight.size(1)
    kernel_size = 40
    weight = torch.randn(in_channels, group_size_out, kernel_size, device="cuda", dtype=torch.float32)
    
    # Create a simple input (batch_size=2, input_width=50)
    x = torch.randn(2, in_channels, 50, device="cuda", dtype=torch.float32)
    
    mod = build_kernel()
    
    with pytest.raises(RuntimeError, match="Weight size exceeds constant memory limit"):
        # Calling forward should trigger a TORCH_CHECK failure from cudaMemcpyToSymbol call.
        mod.forward(x, weight, None, 1, 0, 0, 1)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    # Issue 2: Test that the kernel enforces contiguous input requirement.
    in_channels = 8
    group_size_out = 4
    kernel_size = 3  # small kernel to avoid hitting the constant memory size limit
    weight = torch.randn(in_channels, group_size_out, kernel_size, device="cuda", dtype=torch.float32)

    # Create an input tensor and then make it non contiguous.
    x = torch.randn(2, in_channels, 50, device="cuda", dtype=torch.float32)
    # For example, a transpose makes it non contiguous.
    x_noncontiguous = x.transpose(1, 2)

    mod = build_kernel()
    
    with pytest.raises(RuntimeError, match="must be contiguous"):
        # CHECK_INPUT on x_noncontiguous should cause an error.
        mod.forward(x_noncontiguous, weight, None, 1, 0, 0, 1)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_error_after_kernel_launch(monkeypatch):
    # Issue 3: Test that errors from the kernel launch are not silently ignored.
    # Here we simulate an asynchronous CUDA error by monkeypatching cudaMemcpyToSymbol to force an error.
    # Since we cannot easily force an error in kernel launch, we simulate an error in copying the weight.
    import ctypes

    # Save the original cudaMemcpyToSymbol so we can call it later if needed.
    original_cudaMemcpyToSymbol = torch.cuda._C.cudart().cudaMemcpyToSymbol
    def fake_cudaMemcpyToSymbol(*args, **kwargs):
        # Force an error by returning a non-success error code.
        return 1  # non-zero indicates error in CUDA
    monkeypatch.setattr(torch.cuda._C.cudart(), 'cudaMemcpyToSymbol', fake_cudaMemcpyToSymbol)

    in_channels = 8
    group_size_out = 4
    kernel_size = 3
    weight = torch.randn(in_channels, group_size_out, kernel_size, device="cuda", dtype=torch.float32)
    x = torch.randn(2, in_channels, 50, device="cuda", dtype=torch.float32)
    mod = build_kernel()

    with pytest.raises(RuntimeError):
        # This should raise an error because the fake cudaMemcpyToSymbol fails.
        mod.forward(x, weight, None, 1, 0, 0, 1)
    
    # Restore the original function to avoid side effects
    monkeypatch.undo()
