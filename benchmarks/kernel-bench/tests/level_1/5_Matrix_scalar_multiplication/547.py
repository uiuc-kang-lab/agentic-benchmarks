
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

# Test 1: Pass a tensor with a dtype other than float32 (e.g. float64) triggering issue #1.
def test_dtype_issue():
    my_module = build_kernel()
    A = torch.randn(1024, 1024, dtype=torch.float64, device="cuda")
    s = 3.14
    with pytest.raises(RuntimeError, match="Input tensor A must be of type float."):
        # This should raise a TORCH_CHECK error since A.scalar_type() != torch::kFloat.
        my_module.forward(A, s)

# Test 2: Pass a CPU tensor to trigger the CUDA device check.
def test_cpu_tensor_issue():
    my_module = build_kernel()
    A = torch.randn(1024, 1024, device="cpu", dtype=torch.float32)
    s = 3.14
    with pytest.raises(RuntimeError, match="Input tensor A must be a CUDA tensor."):
        my_module.forward(A, s)

# Test 3: Pass a non-contiguous tensor to trigger issue #2.
# The kernel incorrectly assumes contiguous memory so the result of the multiplication will be wrong.
def test_non_contiguous_issue():
    my_module = build_kernel()
    # Create a contiguous tensor and then create a non-contiguous view (e.g., transpose)
    A = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    A_t = A.transpose(0, 1)  # This tensor is non contiguous.
    s = 2.0
    # Expected correct result computed by PyTorch using elementwise multiplication on the non-contiguous tensor.
    expected = A_t * s
    # The kernel uses A_t.data_ptr() without considering strides.
    result = my_module.forward(A_t, s)
    torch.cuda.synchronize()
    # Since the kernel performs plain elementwise computation over the raw memory,
    # the result will not match the expected value.
    assert not torch.allclose(result, expected, atol=1e-5), "Kernel unexpectedly handled non-contiguous input correctly."

# Note: We do not have a direct test case for the unused 'alignedThreads' or the lack of kernel launch error checking,
# since these are more design/maintenance issues rather than ones that directly trigger a runtime failure.
