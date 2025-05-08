
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 test: Pass an input of integer type.
# Since the kernel is dispatched via AT_DISPATCH_FLOATING_TYPES,
# trying to use a non-floating tensor should trigger an error.
def test_integer_tensor_input():
    cuda_module = build_kernel()

    # Create an integer tensor on CUDA.
    x_int = torch.randint(0, 10, (16, 32, 128, 128), dtype=torch.int32, device="cuda")
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 3
    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel dispatch only supports floats.
        _ = cuda_module.forward(x_int, kernel_size, stride, padding, dilation)

# Issue 2 test: Supply a non-contiguous tensor.
# Non-contiguous tensors may not meet alignment requirements for __ldg().
def test_noncontiguous_tensor_input():
    cuda_module = build_kernel()

    # Create a contiguous tensor and then make it non-contiguous by a transpose operation.
    x = torch.randn(16, 32, 128, 128, device="cuda", dtype=torch.float32)
    x_noncontig = x.transpose(1, 2)  # transpose makes it non-contiguous

    # Use the same parameters.
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 3

    # Run the CUDA kernel.
    out = cuda_module.forward(x_noncontig, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    # For comparison, use PyTorch's own max_pool2d which correctly handles non-contiguous input.
    ref_out = torch.nn.functional.max_pool2d(x_noncontig, kernel_size, stride=stride,
                                              padding=padding, dilation=dilation)
    # The outputs likely differ because __ldg expects aligned data; we check for a significant difference.
    diff = (out - ref_out).abs().max().item()
    assert diff > 1e-3, "Non-contiguous input did not trigger noticeable error; __ldg issue might not be exposed."

