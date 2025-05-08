
import torch
import pytest
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

@pytest.fixture(scope="module")
def cuda_module():
    return build_kernel()

def test_double_precision_issue(cuda_module):
    # Create a double precision tensor.
    x = torch.randn(1024, dtype=torch.double, device="cuda")
    # Get kernel output and reference output.
    kernel_output = cuda_module.forward(x)
    ref_output = torch.sigmoid(x)
    # Due to conversion to float inside the kernel,
    # the maximum absolute difference will be significant.
    diff = torch.abs(kernel_output - ref_output).max().item()
    assert diff > 1e-6, (
        f"Kernel did not exhibit the expected precision loss for double dtype. "
        f"Difference: {diff}"
    )

def test_shared_memory_dtype_issue(cuda_module):
    # Create a half precision tensor.
    x = torch.randn(1024, dtype=torch.float16, device="cuda")
    kernel_output = cuda_module.forward(x)
    ref_output = torch.sigmoid(x)
    diff = torch.abs(kernel_output - ref_output).max().item()
    assert diff > 1e-3, (
        f"Kernel did not exhibit the expected error for half dtype "
        f"(shared memory always as float). Difference: {diff}"
    )

def test_noncontiguous_input_issue(cuda_module):
    # Create a contiguous tensor then make it non-contiguous by transposing.
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # This makes the tensor non-contiguous.
    kernel_output = cuda_module.forward(x_noncontig)
    ref_output = torch.sigmoid(x_noncontig)
    diff = torch.abs(kernel_output - ref_output).max().item()
    assert diff > 1e-6, (
        f"Kernel did not fail as expected when a non-contiguous tensor was provided. "
        f"Difference: {diff}"
    )
