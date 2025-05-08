
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to compile the cuda kernel from kernel.cu
def build_kernel():
    cuda_module = load(
        name="rms_norm_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Reference function for RMSNorm as in the PyTorch Model above.
def rms_norm_reference(x: torch.Tensor, eps: float) -> torch.Tensor:
    # Compute RMS along the feature dimension (dim=1)
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    return x / rms

# Issue 1. Non-contiguous input handling:
# This test creates a non-contiguous input tensor. Since the kernel assumes a contiguous layout,
# the output will differ from the reference.
def test_non_contiguous_input():
    batch_size, features, dim1, dim2 = 4, 8, 16, 16
    eps = 1e-5
    # Create a contiguous tensor and then generate a non-contiguous view
    x_contig = torch.randn(batch_size, features, dim1, dim2, device='cuda')
    # Create a non-contiguous tensor by transposing the last two dims (without calling .contiguous())
    x_noncontig = x_contig.transpose(2, 3)
    # We need the layout as (B, F, ..); here we put it back into the expected shape but non-contiguous:
    x_test = x_noncontig.transpose(2, 3)  # This restores original size but keeps non-contiguity
    assert not x_test.is_contiguous(), "The test input is unexpectedly contiguous."

    my_module = build_kernel()
    # Call the CUDA kernel
    y_kernel = my_module.forward(x_test, eps)
    y_ref = rms_norm_reference(x_test, eps)
    # The outputs are expected to differ because the kernel indexing assumes contiguous memory.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-4), (
        "Kernel output matches reference even though input is non-contiguous. "
        "This indicates the kernel is not handling non-contiguous inputs as expected."
    )

# Issue 2. Invalid input tensor type:
# Passing an integer tensor (which is not a supported floating point type) should trigger an error.
def test_invalid_input_tensor_type():
    batch_size, features, dim1, dim2 = 4, 8, 16, 16
    eps = 1e-5
    # Create an integer tensor
    x_int = torch.randint(0, 10, (batch_size, features, dim1, dim2), device='cuda', dtype=torch.int32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The AT_DISPATCH_FLOATING_TYPES_AND_HALF macro will not pick an implementation for int32,
        # so this should raise an error.
        _ = my_module.forward(x_int, eps)

# Issue 3. Redundant memory loads / performance issues:
# Even if the kernel produces numerically correct results for contiguous valid inputs,
# using a large feature dimension will force the kernel to perform two sequential loops over the features.
# This test uses a large number of features to trigger performance limitations.
def test_large_feature_dimension():
    batch_size, features, dim1, dim2 = 2, 1024, 16, 16  # Large feature dimension to stress the inner loops.
    eps = 1e-5
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    y_kernel = my_module.forward(x, eps)
    y_ref = rms_norm_reference(x, eps)
    # In this test we only check that the outputs are numerically similar.
    # But on profiling one might observe that the kernel is inefficient due to redundant loads.
    assert torch.allclose(y_kernel, y_ref, atol=1e-4), (
        "Kernel output differs from reference output in the large feature dimension case."
    )

# Issue 4. Half-precision handling:
# For half precision inputs, the use of sqrt without special handling may produce reduced precision or
# unexpected behavior. We force a half precision input to see if the result deviates from the reference.
def test_half_precision():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    batch_size, features, dim1, dim2 = 4, 8, 16, 16
    eps = 1e-5
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda', dtype=torch.float16)
    my_module = build_kernel()
    y_kernel = my_module.forward(x, eps)
    y_ref = rms_norm_reference(x, eps)
    # For half precision, allow a larger error tolerance.
    # If the sqrt call is not ideal for __half arithmetic, there could be observable discrepancies.
    assert torch.allclose(y_kernel, y_ref, atol=1e-2), (
        "Kernel output in half precision mode deviates more than expected from the reference."
    )

# Issue 5. Lack of kernel launch error checking:
# This test attempts to trigger a kernel launch error by providing an input of zero elements in the "numel_per_batch"
# dimension. While this is artificial, it checks that the module does not catch kernel launch errors.
def test_zero_numel_per_batch():
    batch_size, features = 4, 8
    eps = 1e-5
    # Create a tensor that has no elements beyond the first two dimensions.
    # According to the kernel, numel_per_batch will be computed by multiplying dimensions >=2.
    # In this case, it will be 1 (not 0), so to force a zero we simulate a scenario by creating an empty tensor.
    x_empty = torch.empty((batch_size, features, 0), device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    # The kernel should essentially do nothing, but the launch parameters will operate on 0 elements.
    # Here we simply check that no error is thrown.
    y_kernel = my_module.forward(x_empty, eps)
    assert y_kernel.numel() == 0, "Expected an empty output tensor when given an empty input."

