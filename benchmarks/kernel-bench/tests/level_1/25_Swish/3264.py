
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="swish_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: The kernel only supports float32 but does not check the type.
def test_input_tensor_type():
    my_module = build_kernel()
    # Create a tensor of float64 on CUDA.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # The kernel does not check type so it will interpret the memory incorrectly.
    # We expect the results to differ significantly from the true swish.
    with pytest.raises(RuntimeError):
        # Using forward should trigger an error or at least produce a wrong result.
        # Since the kernel call might not raise an error immediately, we validate by checking correctness.
        y = my_module.forward(x)
        torch.cuda.synchronize()
        # Compute reference result using PyTorch
        y_ref = x * torch.sigmoid(x)
        # They should not be equal. If they are, something is wrong with type handling.
        assert not torch.allclose(y, y_ref, atol=1e-4), "Kernel incorrectly accepted non-float32 tensor."

# Issue 2: No error checking after the kernel launch. A kernel launch error (like providing a CPU tensor)
# should be caught by our host code.
def test_input_not_cuda():
    my_module = build_kernel()
    # Create a CPU tensor.
    x = torch.randn(1024, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # The TORCH_CHECK in C++ should throw an error if tensor is not on CUDA.
        _ = my_module.forward(x)

# Issue 3: The kernel assumes contiguous input. Test with a non-contiguous tensor.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor then make it non-contiguous by transposing
    x = torch.randn(32, 32, dtype=torch.float32, device="cuda")
    non_contig_x = x.t()  # this transpose makes the tensor non-contiguous
    # Even though some operations in PyTorch can handle non-contiguous memory,
    # our kernel directly uses data_ptr which is valid only for contiguous tensors.
    # We expect either incorrect results or an error.
    y = my_module.forward(non_contig_x)
    torch.cuda.synchronize()
    y_ref = non_contig_x * torch.sigmoid(non_contig_x)
    # Because of the assumed contiguity, the kernel output may not match the reference.
    assert not torch.allclose(y, y_ref, atol=1e-4), "Kernel output unexpectedly matches reference for non-contiguous input!"
