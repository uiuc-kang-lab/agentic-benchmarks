
import pytest
import torch
from torch.nn.functional import conv_transpose2d
from torch.utils.cpp_extension import load

def build_kernel():
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=True,
    )
    return module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_issue_unused_kernel():
    # Issue 1: The custom CUDA kernel is never invoked (forward simply calls at::conv_transpose2d)
    # We compare the extension.forward output with that of torch.nn.functional.conv_transpose2d.
    cuda_mod = build_kernel()
    batch, in_channels, H, W = 2, 3, 8, 8
    # Create a weight tensor in the conventional shape for ConvTranspose2d.
    weight = torch.randn(in_channels, 4, 3, 3, device="cuda", dtype=torch.float32)
    x = torch.randn(batch, in_channels, H, W, device="cuda", dtype=torch.float32)
    # The extension.forward calls at::conv_transpose2d so its output is identical to torch’s.
    out_ext = cuda_mod.forward(x, weight)
    out_ref = conv_transpose2d(x, weight, None, (1, 1), (0, 0), (0, 0), 1, (1, 1))
    assert torch.allclose(
        out_ext, out_ref, atol=1e-5
    ), "Forward output matches built-in conv_transpose2d. Custom kernel was never invoked. - Issue 1"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_issue_incorrect_thread_indexing():
    # Issue 2: The kernel uses threadIdx.x as an output channel indicator and then loops over batches/spatial dims.
    # This is not scalable and general. We simulate the scenario using a high channel count and spatial dimensions
    # and simply check that the output shape is correct (the kernel would create severe problems if thread mapping failed).
    cuda_mod = build_kernel()
    batch, in_channels, H, W = 1, 64, 16, 16
    weight = torch.randn(in_channels, 64, 3, 3, device="cuda", dtype=torch.float32)
    x = torch.randn(batch, in_channels, H, W, device="cuda", dtype=torch.float32)
    out_ext = cuda_mod.forward(x, weight)
    out_ref = conv_transpose2d(x, weight, None, (1, 1), (0, 0), (0, 0), 1, (1, 1))
    # While output shapes match, poor thread indexing would eventually compromise performance or correctness
    # in a custom kernel. Here we check at least that the output dimensions are as expected.
    assert out_ext.shape == out_ref.shape, "Output shape mismatch may indicate incorrect thread indexing - Issue 2"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_issue_input_not_used():
    # Issue 3: The custom kernel omits using the input tensor.
    # If the kernel were used, changing x would not change the output.
    # Here, we call forward on two different inputs and assert that the outputs must differ.
    cuda_mod = build_kernel()
    batch, in_channels, H, W = 1, 3, 8, 8
    weight = torch.randn(in_channels, 4, 3, 3, device="cuda", dtype=torch.float32)
    x1 = torch.randn(batch, in_channels, H, W, device="cuda", dtype=torch.float32)
    x2 = x1 + 1.0  # modify input
    out1 = cuda_mod.forward(x1, weight)
    out2 = cuda_mod.forward(x2, weight)
    # If the output were computed solely from weight (ignoring x), these two would be identical.
    assert not torch.allclose(
        out1, out2, atol=1e-5
    ), "Output is invariant to input changes, suggesting the input tensor is not used - Issue 3"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_issue_constant_memory_overflow():
    # Issue 4: The kernel loads weights into constant memory with a fixed size of 1024 elements.
    # Here we create a weight tensor that exceeds 1024 elements.
    cuda_mod = build_kernel()
    batch, in_channels, H, W = 1, 3, 8, 8
    # For example, weight shape (3, 3, 21, 21) has 3*3*21*21 = 3969 elements, which is > 1024.
    weight = torch.randn(3, 3, 21, 21, device="cuda", dtype=torch.float32)
    x = torch.randn(batch, 3, H, W, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # We expect that trying to load too many elements into the fixed constant memory throws an error.
        out = cuda_mod.forward(x, weight)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_issue_bias_ignored():
    # Issue 5: The custom kernel ignores the bias parameter.
    # We compare the outputs with and without a nonzero bias.
    cuda_mod = build_kernel()
    batch, in_channels, H, W = 1, 3, 8, 8
    weight = torch.randn(3, 4, 3, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(4, device="cuda", dtype=torch.float32)
    x = torch.randn(batch, 3, H, W, device="cuda", dtype=torch.float32)
    out_with_bias = cuda_mod.forward(x, weight, bias)
    out_without_bias = cuda_mod.forward(x, weight)
    # In a correct implementation the bias would affect the result.
    assert not torch.allclose(
        out_with_bias, out_without_bias, atol=1e-5
    ), "Bias appears to be ignored in the forward pass - Issue 5"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_issue_no_cuda_error_checking():
    # Issue 6: There is no error checking for CUDA API calls like cudaMemcpyToSymbol.
    # We simulate a potential failure by providing a weight tensor that is not contiguous.
    cuda_mod = build_kernel()
    batch, in_channels, H, W = 1, 3, 8, 8
    # Force non-contiguous weight tensor by transposing.
    weight = torch.randn(3, 4, 3, 3, device="cuda", dtype=torch.float32).t()
    x = torch.randn(batch, 3, H, W, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        out = cuda_mod.forward(x, weight)
        torch.cuda.synchronize()
