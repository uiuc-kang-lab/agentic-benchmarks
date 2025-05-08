
import torch
import pytest
from torch.utils.cpp_extension import load

def load_kernel():
    cuda_module = load(
        name="selu_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_misaligned_memory():
    # Create a misaligned tensor for float32.
    # We allocate a raw byte tensor with an extra byte,
    # then create a view that is forced to be misaligned.
    batch_size = 16
    dim = 16384
    numel = batch_size * dim
    total_bytes = numel * 4 + 1  # 4 bytes per float32 plus one extra byte

    raw = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
    # Offset by one byte to force misalignment.
    misaligned_bytes = raw.narrow(0, 1, total_bytes - 1)
    # Convert the misaligned byte buffer to float32.
    # Note: frombuffer is not available, so we use view.
    # We must ensure that the total number of bytes is divisible by 4.
    float_data = misaligned_bytes.view(torch.float32)
    # Reshape to the desired shape.
    misaligned_tensor = float_data.view(batch_size, dim)

    # Fill the tensor with random data as expected by the SELU kernel
    misaligned_tensor = misaligned_tensor.clone().contiguous()

    cuda_module = load_kernel()
    # Since the kernel forces vectorized memory loads/stores,
    # this misaligned tensor may trigger undefined behavior or a runtime fault.
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(misaligned_tensor)
        # Force synchronization to capture any asynchronous errors.
        torch.cuda.synchronize()

def test_cpu_tensor_error():
    # The kernel checks that the input tensor is on CUDA.
    # Passing a CPU tensor should trigger a runtime error.
    cpu_tensor = torch.randn(16, 16384, device="cpu", dtype=torch.float32)
    cuda_module = load_kernel()
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor"):
        out = cuda_module.forward(cpu_tensor)
