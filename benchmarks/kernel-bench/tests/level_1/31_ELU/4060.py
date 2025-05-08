
import pytest
import torch
import torch.nn.functional as F
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

# Test 1: Passing a tensor with a non-float32 dtype should trigger an error.
# Since the kernel does not check the dtype, using double (float64) should lead to a runtime error.
def test_incorrect_dtype():
    my_module = build_kernel()
    # Create a double tensor on CUDA.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # The kernel call should fail because the CHECK_INPUT macros don't check dtype,
        # and then the reinterpret_cast<float*>(x.data_ptr()) is invalid.
        my_module.forward(x, 1.0)  # Passing alpha=1.0

# Test 2: Passing a misaligned tensor.
# Create a tensor that is contiguous but whose data pointer is offset from a 16-byte boundary.
# We can do this by allocating an extra element and taking a narrow view.
def test_misaligned_tensor():
    my_module = build_kernel()
    # Allocate a tensor with one extra element.
    base = torch.randn(1025, device="cuda", dtype=torch.float32)
    # Create a view that starts at element 1. Although the view is contiguous,
    # its underlying storage pointer is offset (likely misaligned for float4 loads).
    x = base.narrow(0, 1, 1024)  # 1024 is divisible by 4 but starting offset is 1 float.
    # Compute output from the kernel.
    y_kernel = my_module.forward(x, 1.0)
    # Compute reference ELU using PyTorch.
    y_ref = F.elu(x, alpha=1.0)
    # The results should be close. However, if misaligned loads cause incorrect behavior,
    # this test may fail.
    torch.cuda.synchronize()
    assert torch.allclose(y_kernel, y_ref, atol=1e-5), \
        f"Kernel output does not match reference result for misaligned tensor!"

# Test 3: Passing a CPU tensor should trigger the CHECK_CUDA error.
def test_non_cuda_tensor():
    my_module = build_kernel()
    x = torch.randn(1024, device="cpu", dtype=torch.float32)  # CPU tensor
    with pytest.raises(RuntimeError):
        my_module.forward(x, 1.0)

if __name__ == "__main__":
    pytest.main([__file__])
