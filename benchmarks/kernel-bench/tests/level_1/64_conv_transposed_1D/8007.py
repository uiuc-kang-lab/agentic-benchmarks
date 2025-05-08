
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build and load the CUDA extension
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Fixture to build the kernel module once
@pytest.fixture(scope="session")
def cuda_kernel():
    return build_kernel()

# A helper function to compute a reference conv_transpose1d result using PyTorch
def ref_conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups):
    return torch.conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups)

# Issue 1: The constant memory copy is redundant since the weight in d_weights is never used.
# We simulate this by comparing the output of our kernel with the reference.
def test_redundant_constant_memory(cuda_kernel):
    batch_size = 2
    in_channels = 4
    out_channels = 3
    kernel_size = 3
    length = 16
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    # Create random input and weight tensors (float32)
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Call kernel and reference conv_transpose1d
    y_kernel = cuda_kernel.forward(x, weight, bias, stride, padding, output_padding, groups)
    y_ref = ref_conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups)

    # Since d_weights is not used in computation, the outputs must be identical.
    assert torch.allclose(y_kernel, y_ref, atol=1e-5), \
           "Output mismatch indicates that the redundant constant memory copy impacted the computation."

# Issue 2: The constant memory d_weights is fixed at size 1024 floats;
# using a weight tensor with more than 1024 elements may trigger an overflow.
def test_constant_memory_overflow(cuda_kernel):
    batch_size = 2
    in_channels = 16  # Choose dimensions such that total weight elements > 1024
    out_channels = 16
    kernel_size = 5
    # in_channels * (out_channels) * kernel_size = 16*16*5 = 1280 > 1024
    length = 32
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    # We don't necessarily care about bias here
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Expect the cudaMemcpyToSymbol to be writing beyond allocated constant memory.
    # This may not always crash immediately, but could yield an incorrect result.
    y_kernel = cuda_kernel.forward(x, weight, bias, stride, padding, output_padding, groups)
    y_ref = ref_conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups)

    # The outputs could differ due to memory corruption.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), \
           "Using a weight tensor exceeding the constant memory size should result in a memory error."

# Issue 3: The kernel assumes weight is float32. Passing a weight of a different dtype should cause an error.
def test_weight_wrong_dtype(cuda_kernel):
    batch_size = 2
    in_channels = 4
    out_channels = 3
    kernel_size = 3
    length = 16
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    # Create weight tensor in double precision, which should trigger an error during cudaMemcpyToSymbol
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float64)
    bias = None

    with pytest.raises(RuntimeError):
        cuda_kernel.forward(x, weight, bias, stride, padding, output_padding, groups)

# Issue 4: There is no error capture for cudaMemcpyToSymbol.
# We simulate this by monkeypatching cudaMemcpyToSymbol to force an error.
def test_cudaMemcpyToSymbol_error(monkeypatch, cuda_kernel):
    import ctypes

    # Create a fake cudaMemcpyToSymbol that always returns an error code
    def fake_cudaMemcpyToSymbol(symbol, src, count, offset=0, kind=cuda_runtime.cudaMemcpyHostToDevice):
        return 1  # non-zero indicates an error

    # Monkey-patch the cudaMemcpyToSymbol in the kernel module (this is conceptual as actual patching would require a more involved setup)
    # Here, we simulate by checking that our kernel forward function does not catch errors from the cudaMemcpyToSymbol call.
    # Since our extension is compiled C++ code, we can't directly override cudaMemcpyToSymbol from Python.
    # Instead, we indicate that this test serves as a placeholder for verifying proper error handling in a real scenario.
    pytest.skip("Cannot monkeypatch cudaMemcpyToSymbol in compiled C++; review error checking in kernel code.")
    
# Issue 5: Non-contiguous inputs.
def test_non_contiguous_input(cuda_kernel):
    batch_size = 2
    in_channels = 4
    out_channels = 3
    kernel_size = 3
    length = 16
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    # Create contiguous input and then make it non-contiguous by transposing (if applicable)
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    x_non_contig = x.transpose(1, 2)  # This makes the tensor non-contiguous
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    with pytest.raises(RuntimeError):
        cuda_kernel.forward(x_non_contig, weight, bias, stride, padding, output_padding, groups)
