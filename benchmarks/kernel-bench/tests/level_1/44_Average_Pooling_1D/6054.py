
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="pooling_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger issue by providing an input tensor that is not float32.
def test_input_tensor_type():
    cuda_module = build_kernel()
    # Create a double precision tensor (should be float64) that the kernel is not handling.
    x = torch.randn(2, 4, 16, dtype=torch.float64, device="cuda")
    kernel_size = 3
    stride = 2
    padding = 1
    with pytest.raises(RuntimeError):
        # The kernel expects float input, otherwise an error or incorrect behavior may occur.
        cuda_module.forward(x, kernel_size, stride, padding)

# Test case 2: Trigger issue with shared memory mis-allocation.
def test_shared_memory_misallocation():
    cuda_module = build_kernel()
    # Choosing parameters such that the computed tile boundaries in the kernel become non-trivial.
    batch_size = 2
    in_channels = 3
    input_length = 25  # small input length that stresses the tile boundary computation.
    kernel_size = 5
    stride = 3
    padding = 2
    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32, device="cuda")
    # Compute expected output length manually as in kernel: (input_length + 2*padding - kernel_size)//stride + 1
    expected_output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    output = cuda_module.forward(x, kernel_size, stride, padding)
    # For simplicity we re-run PyTorch's AvgPool1d as a reference.
    ref_module = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    ref_output = ref_module(x)
    # The error in shared memory size allocation may lead to incorrect results.
    assert not torch.allclose(output, ref_output, atol=1e-4), "Test expected differences due to shared memory misallocation, but the outputs match."

# Test case 3: Trigger issue with the tile boundary computation.
def test_tile_boundary_computation():
    cuda_module = build_kernel()
    # Use parameters where the padded region will require careful boundary handling.
    batch_size = 1
    in_channels = 1
    input_length = 10
    kernel_size = 4
    stride = 3
    padding = 3
    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32, device="cuda")
    output = cuda_module.forward(x, kernel_size, stride, padding)
    # Compare against PyTorch's own AvgPool1d output.
    ref_module = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    ref_output = ref_module(x)
    # In a correct kernel the outputs would match. In our intentionally problematic kernel the boundaries might be computed wrongly.
    assert not torch.allclose(output, ref_output, atol=1e-4), "Test expected differences due to tile boundary miscalculation, but the outputs match."

# Test case 4: Test performance inefficiency by repeatedly calling the forward function.
def test_stream_creation_overhead():
    cuda_module = build_kernel()
    batch_size = 4
    in_channels = 4
    input_length = 64
    kernel_size = 3
    stride = 1
    padding = 1
    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32, device="cuda")
    # Run the forward operation multiple times to highlight that each call creates and destroys a stream.
    # While we cannot measure performance in a unit test, we can check for correct execution over repeated calls.
    outputs = []
    for _ in range(10):
        outputs.append(cuda_module.forward(x, kernel_size, stride, padding))
    # Ensure that none of the outputs are completely zero (an indicator of some internal failure)
    for out in outputs:
        assert out.abs().sum() > 0, "Output should have non-zero values."
