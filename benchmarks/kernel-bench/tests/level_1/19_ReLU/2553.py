
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA module.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: No CUDA error checking.
# To trigger a potential CUDA error we can force an out-of-bound access.
# One way is to create a tensor with a “size” that is inconsistent with the underlying allocation.
# Although it's tricky to force a CUDA error in this simple kernel, we simulate a scenario by
# calling the kernel with an intentionally corrupted input pointer.
def test_cuda_error_checking(monkeypatch):
    # This test is somewhat artificial because the kernel does not check errors.
    # We simulate a condition by creating a dummy input and then freeing it prematurely.
    cuda_module = build_kernel()

    # Create a normal input tensor and an output tensor.
    input_tensor = torch.randn(1024, device='cuda', dtype=torch.float32)
    
    # Corrupt input tensor by wrapping the data_ptr in a dummy tensor that does not share the same allocation.
    # One way is to make a tensor and then use an invalid pointer value.
    # Since we cannot easily simulate a pointer corruption from Python,
    # we check that the function call does not perform error checking by not raising even if something is wrong.
    # (This test is more of a placeholder that reminds us to implement CUDA error checking.)
    try:
        output_tensor = cuda_module.forward(input_tensor)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail("Kernel raised an error, but it should silently fail (lack of error checking): " + str(e))

# Issue 2: Assumption of contiguous memory.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    
    # Create a tensor and intentionally make it non-contiguous by transposing.
    x = torch.randn(32, 64, device='cuda', dtype=torch.float32)
    x_non_contiguous = x.t()  # Transpose makes it non-contiguous.
    # For the purpose of ReLU, expected behavior is to get the same result
    # as torch.relu, but since our kernel uses data_ptr arithmetic without stride,
    # it will treat the memory as if it were contiguous.
    with pytest.raises(AssertionError):
        # We expect a mismatch since the underlying data ordering differs.
        output = cuda_module.forward(x_non_contiguous)
        torch.cuda.synchronize()
        output_ref = torch.relu(x_non_contiguous)
        # The outputs will not match correctly if the kernel misinterprets strides.
        assert torch.allclose(output, output_ref), "Non-contiguous tensor produced an incorrect result."

# Issue 3: Limited type dispatch – unsupported type (e.g., float16).
def test_unsupported_dtype():
    cuda_module = build_kernel()
    
    # Create a half precision tensor (float16) on CUDA.
    x_half = torch.randn(1024, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # The AT_DISPATCH_FLOATING_TYPES macro does not include float16,
        # so the kernel should not support it and is expected to raise.
        output = cuda_module.forward(x_half)
        torch.cuda.synchronize()

# Issue 4: Input tensor on wrong device (CPU instead of CUDA).
def test_cpu_input():
    cuda_module = build_kernel()
    
    # Create a CPU tensor.
    x_cpu = torch.randn(1024, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Since the kernel is a CUDA kernel, passing a CPU tensor should lead to an error.
        output = cuda_module.forward(x_cpu)
        torch.cuda.synchronize()
