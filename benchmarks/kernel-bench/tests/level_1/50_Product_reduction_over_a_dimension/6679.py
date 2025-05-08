
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

# Test 1: Incorrect Indexing Computation
# Use a tensor shape and reduction dimension such that the mapping from the flat output index to the input is nontrivial.
# For example, a 3D tensor where reduction is performed over the middle dimension.
def test_incorrect_indexing():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cuda_module = build_kernel()
    # Create a tensor of shape (4, 3, 5) and reduce along dimension 1.
    x = torch.randn(4, 3, 5, device='cuda', dtype=torch.float32).contiguous()
    dim = 1
    # Run our CUDA kernel
    y_kernel = cuda_module.forward(x, dim)
    # Run PyTorch native product reduction
    y_ref = torch.prod(x, dim=dim)
    # Because of the buggy indexing, the results should differ.
    # We check for a discrepancy.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Test failed to trigger indexing error: kernel output matched expected result "
        "despite incorrect indexing."
    )

# Test 2: Data Type Mismatch (Kernel only supports float32)
# Provide an input tensor of a type other than float32, e.g. float64.
def test_data_type_support():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cuda_module = build_kernel()
    # Create a tensor with type float64 (double)
    x = torch.randn(16, 256, 256, device='cuda', dtype=torch.float64).contiguous()
    dim = 1
    with pytest.raises(RuntimeError):
        # This should fail because kernel expects float32 and uses data_ptr<float>()
        _ = cuda_module.forward(x, dim)

# Test 3: Non-contiguous input tensor
# Provide a tensor that is not contiguous and expect the kernel CHECK_INPUT macro to trigger an error.
def test_non_contiguous_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cuda_module = build_kernel()
    # Create a contiguous tensor and then perform a transpose to make it non-contiguous.
    x = torch.randn(16, 256, 256, device='cuda', dtype=torch.float32)
    # Transpose two dimensions to break contiguity
    x_non_contiguous = x.transpose(1, 2)
    dim = 1  # reduction still along original dim index
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(x_non_contiguous, dim)
        
if __name__ == "__main__":
    pytest.main([__file__])
