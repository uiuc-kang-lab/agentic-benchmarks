
import torch
import pytest
from torch.utils.cpp_extension import load

# Build and load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test for non-contiguous tensors.
# The kernel does not check tensor contiguity. A non-contiguous tensor (for example, a transpose)
# will have an unexpected memory layout and produce incorrect results.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a tensor and then a non-contiguous view (by transposing).
    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float)
    A_noncontig = A.t()  # This tensor is not contiguous.
    s = 2.0
    # We use the kernel even though the tensor is non-contiguous.
    C = my_module.forward(A_noncontig, s)
    # Compute expected output by using a contiguous version.
    C_ref = A_noncontig.contiguous() * s
    # The results may differ because the kernel does not account for the strange memory stride.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Expected kernel multiplication to fail for non-contiguous inputs due to mis-aligned accesses."
    )

# Issue 2: Test for unsupported tensor data type.
# The kernel explicitly checks for torch::kFloat. Passing a tensor with a type other than float should fail.
def test_unsupported_dtype():
    my_module = build_kernel()
    A_double = torch.randn(1024, 1024, device="cuda", dtype=torch.double)
    s = 2.0
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(A_double, s)
    assert "Input tensor A must be of type float" in str(excinfo.value)

# Issue 3: Test that creating and using separate CUDA streams may affect behavior.
# This test sets a default stream and then launches the kernel, checking if the stream difference
# causes any synchronization issues. (In a proper setting, using custom streams can violate assumptions.)
def test_cuda_stream_misuse():
    my_module = build_kernel()
    # Create a contiguous float tensor.
    A = torch.randn(8192, 8192, device="cuda", dtype=torch.float)
    s = 3.0
    # Record the current default stream.
    default_stream = torch.cuda.current_stream().cuda_stream
    # Call the kernel from the default stream.
    C = my_module.forward(A, s)
    # The kernel launches work on two different streams that are independently synchronized.
    # However, if any dependency were assumed on the default stream, it may not hold.
    # We force synchronization on the default stream and then verify the result.
    torch.cuda.synchronize()
    C_ref = A * s
    # In a correct implementation using the current stream, we would have:
    assert torch.allclose(C, C_ref, atol=1e-5), (
        "Kernel output does not match expected result - possible issue with CUDA stream synchronization."
    )

# Issue 4: Test for missing error checking (simulate a failure).
# While error checking in kernel launches is not directly accessible in Python,
# we can try to force an error by providing an invalid (empty) tensor.
def test_kernel_error_launch():
    my_module = build_kernel()
    # Pass an empty tensor (which might lead to zero threads, but our kernel code paths should handle it gracefully).
    # To simulate an error condition, we pass non-CUDA tensor.
    A_cpu = torch.randn(1024, 1024, device="cpu", dtype=torch.float)
    s = 2.0
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(A_cpu, s)
    assert "Input tensor A must be a CUDA tensor." in str(excinfo.value)
    
if __name__ == "__main__":
    pytest.main([__file__])
