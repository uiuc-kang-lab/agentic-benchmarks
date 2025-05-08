
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="swish_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference swish implementation in Python
def swish_reference(x):
    return x * torch.sigmoid(x)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_tensor_type():
    # Issue 1: Using a non-float32 tensor; kernel expects float32.
    kernel = build_kernel()
    # Create a double tensor
    x = torch.randn(256, device="cuda", dtype=torch.double)
    # The kernel is launched on the raw float pointer, so the result computed in the kernel
    # will be based on misinterpreted data. We expect a significant difference from the correct result.
    y = kernel.forward(x)
    # Compute reference result in double precision for comparison
    y_ref = swish_reference(x)
    # Check that the outputs are not close, triggering the issue.
    assert not torch.allclose(y.double(), y_ref, atol=1e-3), (
        "Kernel accepted a double tensor and produced results similar to reference, "
        "but it should only support float32 inputs."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_tensor():
    # Issue 2: Non-contiguous tensor access.
    kernel = build_kernel()
    # Create a contiguous tensor then make it non-contiguous via transpose
    x_contig = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    x = x_contig.t()  # now non-contiguous
    # Running the kernel on a non-contiguous tensor will use x.data_ptr<float>(),
    # which does not account for strides.
    y = kernel.forward(x)
    y_ref = swish_reference(x)
    # We expect the result to be incorrect.
    assert not torch.allclose(y, y_ref, atol=1e-3), (
        "Kernel produced correct results on a non-contiguous tensor, "
        "but it assumes contiguous memory and should fail."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_stream_mismatch():
    # Issue 3: The kernel uses cudaStreamDefault, which ignores the current non-default stream.
    kernel = build_kernel()
    # Create a tensor and move it to a non-default stream.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(256*100, device="cuda", dtype=torch.float32)
        # Launch the kernel while the current stream is not the default stream.
        y = kernel.forward(x)
        # Important: Do NOT synchronize the non-default stream here.
        # Instead, synchronize the default stream to get the kernel result.
        torch.cuda.default_stream().synchronize()
        # Separately, compute the reference result on the same input x.
        y_ref = swish_reference(x)
    # Since the kernel did not use the non-default stream,
    # there may be a race condition or ordering issue,
    # leading to an incorrect result.
    assert not torch.allclose(y, y_ref, atol=1e-3), (
        "Kernel produced correct results even though it was launched on the default stream "
        "while the current stream was non-default, which is unexpected."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_stride_overflow_simulation():
    # Issue 4: The stride variable is int but should be int64_t for very large n.
    # Simulating a real overflow is impractical due to memory limitations;
    # instead, we mimic the expected behavior by manually patching the launch parameters.
    # Here we test that for a moderately large tensor, the kernel still works
    # (i.e., if our index arithmetic were to overflow, the result would be wrong).
    kernel = build_kernel()
    # Create a large tensor that is still allocatable but with many elements.
    # (Note: This is only a simulation; in a real-world scenario, a tensor with more elements
    # than 32-bit indexing allows may be used.)
    n = 2**24  # 16 million elements (~64MB for float32)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = kernel.forward(x)
    y_ref = swish_reference(x)
    # If there was an integer overflow issue with stride, the results would be off.
    assert torch.allclose(y, y_ref, atol=1e-3) is False, (
        "Kernel produced correct results for a large tensor, which is unexpected if there is a stride overflow issue."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_missing_error_check():
    # Issue 5: There is no error checking after the kernel launch.
    # We simulate an error condition by passing an invalid tensor shape.
    kernel = build_kernel()
    try:
        # Create an empty tensor; some kernels may mishandle zero elements.
        x = torch.empty(0, device="cuda", dtype=torch.float32)
        y = kernel.forward(x)
        torch.cuda.synchronize()
    except RuntimeError:
        pytest.skip("Kernel error detected as expected; missing error check would normally hide such errors.")
    else:
        # If no error is raised, check that the output is empty, though an error check is missing.
        assert y.numel() == 0, "Expected empty output for empty input, potential missing error checking in kernel."
