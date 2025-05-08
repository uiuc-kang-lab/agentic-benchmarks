
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

def build_kernel():
    # Loads the CUDA extension from the file kernel.cu
    cuda_module = load(
        name="custom_avg_pool1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_input_dtype_issue():
    # Issue 1: Passing an input tensor with a wrong data type (float64 instead of float32)
    module = build_kernel()
    batch_size, channels, length = 4, 8, 16
    # Create a double tensor (float64).
    x = torch.randn(batch_size, channels, length, dtype=torch.float64, device="cuda")
    kernel_size, stride, padding = 3, 1, 1

    # Expected behavior from PyTorch's AvgPool1d when using float32 input.
    pool = nn.AvgPool1d(kernel_size, stride, padding)
    expected = pool(x.float())
    
    # Run the CUDA kernel with a non-float32 tensor.
    out = module.forward(x, kernel_size, stride, padding)
    
    # The output is expected to be incorrect because the kernel interprets the bits as float.
    # We check that the result is not close to the expected output.
    assert not torch.allclose(out, expected, atol=1e-5), (
        "Kernel incorrectly handled non-float32 input; output matches expected despite wrong dtype."
    )

def test_concurrent_kernel_issue():
    # Issue 2: Launching kernels concurrently with different pooling parameters.
    module = build_kernel()
    batch_size, channels, length = 4, 8, 32
    # Two different inputs.
    x1 = torch.randn(batch_size, channels, length, dtype=torch.float32, device="cuda")
    x2 = torch.randn(batch_size, channels, length, dtype=torch.float32, device="cuda")

    # Use different parameters for each call.
    params1 = (3, 1, 1)  # kernel_size, stride, padding
    params2 = (5, 2, 2)

    # Set up two CUDA streams.
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    # Launch kernels concurrently on different streams.
    with torch.cuda.stream(stream1):
        out1 = module.forward(x1, *params1)
    with torch.cuda.stream(stream2):
        out2 = module.forward(x2, *params2)

    # Synchronize to ensure kernels have finished.
    torch.cuda.synchronize()

    # Compute expected outputs.
    expected1 = nn.AvgPool1d(*params1)(x1)
    expected2 = nn.AvgPool1d(*params2)(x2)

    # Due to constant memory overwrites, one or both outputs may be corrupted.
    # We expect at least one of the outputs not to match its expected result.
    diff1 = (out1 - expected1).abs().max().item()
    diff2 = (out2 - expected2).abs().max().item()

    assert diff1 > 1e-3 or diff2 > 1e-3, (
        "Concurrent kernel launches did not show the expected conflict from constant memory reuse."
    )

def test_cudaMemcpyToSymbol_error_issue(monkeypatch):
    # Issue 3: Lack of error checking after cudaMemcpyToSymbol.
    # It is difficult to force cudaMemcpyToSymbol to fail in a controlled manner;
    # however, we can simulate a failure by monkey-patching cudaMemcpyToSymbol in the extension module.
    # This test is mostly illustrative.
    module = build_kernel()

    # Save the original function.
    import ctypes
    original_memcpy = torch.cuda._C._cuda_memcpy_to_symbol if hasattr(torch.cuda._C, "_cuda_memcpy_to_symbol") else None
    # Define a fake memcpy that always raises an error.
    def fake_cudaMemcpyToSymbol(symbol, src, count):
        raise RuntimeError("Simulated cudaMemcpyToSymbol failure")

    # Monkey-patch the CUDA memcpy call if available.
    if original_memcpy is not None:
        monkeypatch.setattr(torch.cuda._C, "_cuda_memcpy_to_symbol", fake_cudaMemcpyToSymbol)
        batch_size, channels, length = 4, 8, 16
        x = torch.randn(batch_size, channels, length, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="Simulated cudaMemcpyToSymbol failure"):
            module.forward(x, 3, 1, 1)
        # Restore the original function.
        monkeypatch.undo()
    else:
        pytest.skip("Cannot access cudaMemcpyToSymbol in torch.cuda._C; skipping test.")

def test_invalid_output_length_issue():
    # Issue 4: When the pooling parameters lead to a negative output length.
    module = build_kernel()
    batch_size, channels, length = 4, 8, 5  # small input length
    # Choose parameters that force output_length to be negative.
    kernel_size, stride, padding = 7, 1, 0  # (5 + 0 - 7) / 1 + 1 = -1
    x = torch.randn(batch_size, channels, length, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError):
        # This call should fail since it will try to allocate a tensor with a negative size.
        module.forward(x, kernel_size, stride, padding)

def test_non_contiguous_input_issue():
    # Issue 6: The kernel assumes contiguous input without checking.
    module = build_kernel()
    batch_size, channels, length = 4, 8, 32
    x = torch.randn(batch_size, channels, length, dtype=torch.float32, device="cuda")
    # Make the tensor non-contiguous by a simple transpose (swapping channels and length).
    x_non_contiguous = x.transpose(1, 2)
    # PyTorch's AvgPool1d expects input of shape (batch, channels, length),
    # so we make a contiguous copy of the expected result for comparison.
    pool = nn.AvgPool1d(3, 1, 1)
    expected = pool(x_non_contiguous.contiguous())
    out = module.forward(x_non_contiguous, 3, 1, 1)

    # Since the kernel performs a flat indexing assumption, the output will likely be incorrect.
    assert not torch.allclose(out, expected, atol=1e-5), (
        "Kernel produced correct results with non-contiguous input, but it is expected to assume contiguous memory."
    )
