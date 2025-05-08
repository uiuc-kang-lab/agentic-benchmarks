
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

def reference_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    # Compute the HardSigmoid activation as: clamp((x + 3) / 6, 0, 1)
    return ((x + 3) / 6).clamp(0, 1)

def test_non_contiguous_tensor(kernel_module):
    # Issue 1: Non-contiguous tensors
    # Create a contiguous tensor then transpose it to make it non-contiguous.
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x_nc = x.t()  # now non-contiguous
    # Run the CUDA kernel
    y = kernel_module.forward(x_nc)
    # Calculate the expected output using PyTorch's functional implementation.
    y_ref = reference_hardsigmoid(x_nc)
    # Since our kernel assumes contiguous memory, the output will be incorrect.
    # We want this test to detect the issue by checking that the result is NOT equal to the reference.
    assert not torch.allclose(y, y_ref, atol=1e-5), (
        "Kernel unexpectedly produced correct results on a non-contiguous tensor; "
        "the kernel should not work properly in this case."
    )

def test_half_precision_input(kernel_module):
    # Issue 2: The kernel does not support half precision.
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    # Expect that calling the kernel on a half tensor raises an error.
    with pytest.raises(RuntimeError):
        _ = kernel_module.forward(x)

def test_missing_stream_synchronization(kernel_module):
    # Issue 3: Lack of proper stream synchronization.
    # One way to try to trigger an error that might be hidden by asynchronous execution is by
    # deliberately passing an input tensor that is on the wrong device (CPU) so that the kernel
    # launch fails immediately.
    x = torch.randn(1024, dtype=torch.float32)  # CPU tensor
    with pytest.raises(RuntimeError):
        _ = kernel_module.forward(x)

def test_index_overflow_potential(kernel_module):
    # Issue 4: Potential integer overflow with very large tensors.
    # Although it is impractical to allocate a tensor with > 2^31 elements,
    # we simulate the scenario by monkey-patching the numel value (warning: this is for testing purposes only).
    class FakeTensor(torch.Tensor):
        def __new__(cls, orig_tensor, fake_numel):
            return torch.Tensor._make_subclass(orig_tensor, orig_tensor.dtype, orig_tensor.requires_grad)
        def numel(self):
            return 2**31 + 100  # A value beyond the range of int32
    x_real = torch.randn(1024, device="cuda", dtype=torch.float32)
    # Create a fake tensor that wraps x_real but reports a huge numel.
    x_fake = FakeTensor(x_real, fake_numel=2**31 + 100)
    # Because the kernel iterates from 0 to numel (which is far beyond the allocated data),
    # this should lead to an out-of-bounds access and eventually a CUDA error.
    with pytest.raises(RuntimeError):
        _ = kernel_module.forward(x_fake)
