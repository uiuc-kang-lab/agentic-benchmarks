
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to build the extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Pass a CUDA tensor with an unsupported type (int32) to trigger the device code bug.
# Although the wrapper checks for float32/float64, triggering this check will show that the kernel
# was written assuming a max function call that is not device compliant.
def test_invalid_dtype():
    my_module = build_kernel()
    # Create input tensor with invalid dtype (int32) on CUDA.
    x = torch.randint(0, 100, (16, 16384), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError) as excinfo:
        # This should raise an error during runtime from the TORCH_CHECK that restricts the type.
        my_module.forward(x, 1)
    assert "input must be float32 or float64" in str(excinfo.value)

# Test 2: Pass a tensor with an empty reduction dimension (dim_size == 0) to trigger undefined behavior
# from uninitialized shared memory and reduction loops.
def test_empty_reduction_dim():
    my_module = build_kernel()
    # Create a tensor where the softmax dimension is empty.
    x = torch.randn(16, 0, dtype=torch.float32, device="cuda")
    # The expected PyTorch log_softmax of an empty tensor is empty.
    # However, our kernel does not check for dim_size==0 and may result in a crash
    # or produce NaNs. We check that the kernel does not produce valid numbers.
    with pytest.raises(RuntimeError):
        out = my_module.forward(x, 1)
        torch.cuda.synchronize()
