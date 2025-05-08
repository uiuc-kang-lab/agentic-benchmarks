
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu file.
    # It is assumed that kernel.cu is in the current directory.
    cuda_module = load(
        name="log_softmax_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Test 1: Passing a CPU tensor should trigger the TORCH_CHECK that requires a CUDA tensor.
def test_non_cuda_tensor():
    cuda_module = build_kernel()
    x = torch.randn(16, 16384, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="input must be a CUDA tensor"):
        # This call should error out because the input is not on the GPU.
        cuda_module.forward(x, 1)

# Test 2: Passing a non-floating-point tensor should trigger the TORCH_CHECK about scalar type.
def test_non_floating_dtype():
    cuda_module = build_kernel()
    # Create an integer tensor on CUDA.
    x = torch.randint(0, 10, (16, 16384), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="input must be float32 or float64"):
        cuda_module.forward(x, 1)

# Test 3: Compare the kernel output with PyTorch's log_softmax. This test may reveal numerical differences
# due to the redundant reloading of inputs in the final pass.
def test_log_softmax_output_accuracy():
    cuda_module = build_kernel()
    # Create an input with a wide dynamic range to stress numerical precision.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32) * 10.0
    # Compute reference output using PyTorch.
    ref = torch.log_softmax(x, dim=1)
    # Compute output from our CUDA kernel.
    out = cuda_module.forward(x, 1)
    torch.cuda.synchronize()
    # The outputs should be nearly equal. Use a tight tolerance.
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4), \
        f"Kernel output does not match PyTorch reference. Max diff: {(ref - out).abs().max().item()}"

# Test 4: (Optional) Launch configuration test.
# If dim_size is much smaller than the blockDim.x calculated by the kernel,
# the extra threads may be idle yet still participate in the reduction. This test uses a small dim_size.
def test_small_dim_size():
    cuda_module = build_kernel()
    # Setting a small dimension (e.g. 7) to force many threads doing no work.
    # We permute the input accordingly, so our kernel treats the last dimension as softmax dim.
    x = torch.randn(64, 7, device="cuda", dtype=torch.float32)
    ref = torch.log_softmax(x, dim=1)
    out = cuda_module.forward(x, 1)
    torch.cuda.synchronize()
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4), \
        f"Kernel output for small dim_size does not match PyTorch reference. Max diff: {(ref - out).abs().max().item()}"
