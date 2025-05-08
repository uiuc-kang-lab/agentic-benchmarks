
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu file.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports float32. This test will pass a double tensor to trigger an error.
def test_input_type_not_float32():
    cuda_module = build_kernel()
    # Create input tensor with type double (float64) instead of float32.
    x = torch.randn(2, 4, 10, dtype=torch.double, device="cuda")
    # The kernel does not verify type so it may cause a CUDA memory access error.
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, 3, 1, 1, 1, False)

# Issue 2: When return_indices is True, the output tensor is concatenated instead of returning a tuple.
def test_return_indices_format():
    cuda_module = build_kernel()
    # Create a proper float32 input.
    x = torch.randn(2, 4, 10, dtype=torch.float32, device="cuda")
    ret = cuda_module.forward(x, 2, 1, 0, 1, True)
    # The expected behavior for MaxPool1d with return_indices=True is to return a tuple: (output, indices)
    # Here, the kernel incorrectly concatenates them along the last dimension.
    # For a kernel_size of 2, output_length is computed as:
    output_length = ((10 + 0 - 1 * (2 - 1) - 1) // 1) + 1  # = 10 - 1 = 9 (approximately)
    expected_shape = (2, 4, output_length * 2)
    assert ret.shape == expected_shape, (
        f"Return shape {ret.shape} does not match expected concatenated shape {expected_shape}. "
        "This indicates the kernel is incorrectly concatenating output and indices."
    )

# Issue 3: Using __constant__ memory for kernel parameters may cause race conditions.
def test_constant_memory_race_condition():
    cuda_module = build_kernel()
    # Create two different inputs with different kernel parameters.
    x1 = torch.randn(2, 4, 10, dtype=torch.float32, device="cuda")
    x2 = torch.randn(2, 4, 12, dtype=torch.float32, device="cuda")
    
    # Launch kernel in two separate CUDA streams concurrently.
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    out1 = None
    out2 = None
    with torch.cuda.stream(stream1):
        # Using one set of kernel parameters.
        out1 = cuda_module.forward(x1, 3, 1, 1, 1, False)
    with torch.cuda.stream(stream2):
        # Using a different input size which changes input_length and hence kernel parameters.
        out2 = cuda_module.forward(x2, 3, 1, 1, 1, False)
    torch.cuda.synchronize()
    
    # The expected output lengths are computed from the individual inputs.
    out1_expected_len = ((10 + 2 - 1 * (3 - 1) - 1) // 1) + 1  # input_length=10, padding=1
    out2_expected_len = ((12 + 2 - 1 * (3 - 1) - 1) // 1) + 1  # input_length=12, padding=1
    
    assert out1.shape == (2, 4, out1_expected_len), (
        f"Output shape for first input: {out1.shape} does not match expected shape (2,4,{out1_expected_len})."
    )
    assert out2.shape == (2, 4, out2_expected_len), (
        f"Output shape for second input: {out2.shape} does not match expected shape (2,4,{out2_expected_len})."
    )
    # Note: If constant memory is overwritten concurrently, one of the outputs may be computed with wrong parameters.

# Issue 4: No error checking after CUDA API calls. We simulate a failure by passing an input with an invalid length.
def test_invalid_kernel_params():
    cuda_module = build_kernel()
    # Create an input tensor with sequence length 0, which should cause a failure in output_length computation.
    x = torch.randn(2, 4, 0, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, 3, 1, 1, 1, False)
