
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the extension module from the source file "kernel.cu"
def build_kernel():
    cuda_module = load(
        name="hardsigmoid_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: FP16 (half precision) is not supported.
def test_fp16_input_failure():
    my_module = build_kernel()
    # Create a half precision tensor on CUDA.
    x = torch.randn(16, 16384, device='cuda', dtype=torch.half)
    # The kernel dispatch only supports float and double.
    with pytest.raises(RuntimeError) as excinfo:
        # This should raise an error because hardsigmoid_cuda does not support half.
        my_module.forward(x)
    # Optionally, check that the error message complains about unsupported scalar type.
    assert "not implemented" in str(excinfo.value).lower()

# Issue 2: Noncontiguous input. The kernel assumes contiguous memory.
def test_noncontiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor.
    x = torch.randn(32, 513, device='cuda')  # purposely choose shape not a multiple of common block sizes
    # Create a noncontiguous tensor by transposing.
    x_noncontig = x.t()  
    # Compute expected result using PyTorch's built-in function.
    expected = torch.nn.functional.hardsigmoid(x_noncontig)
    # Run the custom CUDA kernel.
    output = my_module.forward(x_noncontig)
    torch.cuda.synchronize()
    # The custom kernel iterates over the flattened tensor assuming contiguous layout.
    # Thus, if the input is noncontiguous, the result will be wrong.
    # We check that the results are not equal.
    assert not torch.allclose(output, expected), "Kernel produced correct results on noncontiguous input, but it should fail."

# Issue 3: Index arithmetic overflow if the number of elements is very large.
# This test artificially monkey-patches numel() to simulate an overflow scenario.
def test_index_overflow(monkeypatch):
    my_module = build_kernel()
    # Create a small tensor so that memory allocation is safe.
    x = torch.randn(16, 16384, device='cuda')
    # Save the original numel function.
    orig_numel = x.numel

    # Monkey-patch numel() to return a huge number to simulate overflow in index arithmetic.
    def fake_numel():
        return 2**31 + 100  # larger than max int

    monkeypatch.setattr(x, 'numel', fake_numel)
    # When the kernel is launched, it will iterate as though there are far more elements,
    # reading out-of-bounds memory. We expect the kernel to eventually catch a CUDA error.
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(x)
        # Synchronize to catch asynchronous errors.
        torch.cuda.synchronize()
    err_msg = str(excinfo.value).lower()
    # Check that the error seems related to memory or launch failure.
    assert "cuda" in err_msg or "illegal" in err_msg or "segmentation" in err_msg
    # Restore original numel (good practice though not strictly needed here)
    monkeypatch.undo()
