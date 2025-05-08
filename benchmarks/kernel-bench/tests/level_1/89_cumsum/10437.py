
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="cumsum_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# Issue 1: Kernel is hard-coded for float32; using float64 will yield incorrect results.
def test_input_tensor_type():
    cuda_module = build_kernel()
    # Create a float64 (double precision) tensor.
    x = torch.randn(128, 4000, device='cuda', dtype=torch.float64)
    # Call the kernel (which interprets the input as float32).
    result = cuda_module.forward(x, 1)
    torch.cuda.synchronize()
    expected = torch.cumsum(x, dim=1)
    # Because the kernel misinterprets the memory, the result will likely differ.
    with pytest.raises(AssertionError):
        assert torch.allclose(result, expected, atol=1e-5), (
            "Kernel output unexpectedly matches torch.cumsum for float64 input. "
            "This demonstrates the hard-coded type issue."
        )

# Issue 2: Kernel requires contiguous input. Passing in a non-contiguous tensor should fail.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    x = torch.randn(128, 4000, device='cuda', dtype=torch.float32)
    # Create a non-contiguous tensor by slicing.
    x_noncontig = x[:, ::2]
    with pytest.raises(RuntimeError):
        # The CHECK_INPUT macro should trigger an error because x_noncontig is not contiguous.
        cuda_module.forward(x_noncontig, 1)

# Issue 3: Kernel assumes a particular memory layout. When given a tensor whose logical layout does not match 
# the assumed outer/stride/inner ordering, the cumulative sum result will be incorrect.
def test_memory_layout_issue():
    cuda_module = build_kernel()
    # Create a 3D tensor.
    x = torch.randn(16, 32, 4, device='cuda', dtype=torch.float32)
    # Permute dimensions to change the logical layout.
    x_permuted = x.permute(1, 0, 2)
    # Do not call .contiguous() so that the tensor remains non–standard.
    # (The kernel requires contiguity; if forced contiguous by clone(), the data reordering would nullify the test.)
    with pytest.raises(RuntimeError):
        # The non–contiguous permuted tensor should trigger the CHECK_INPUT, or even if it passed the contiguity check
        # by accident, the computed indices will be wrong compared to torch.cumsum.
        out = cuda_module.forward(x_permuted, 1)
        torch.cuda.synchronize()
        expected = torch.cumsum(x_permuted, dim=1)
        # For testing we assume a mismatch will occur.
        assert torch.allclose(out, expected, atol=1e-5), (
            "Kernel output unexpectedly matches expected cumsum for a tensor with non–standard layout."
        )

# Issue 4: The kernel does not perform error checking after launch. 
# One can simulate this by forcing a situation where the kernel should error (e.g., an extremely large tensor)
# but no immediate error is reported, leading to silent failure.
def test_no_kernel_launch_error_checking():
    cuda_module = build_kernel()
    # Attempt to create a tensor with an extremely large size that might trigger index calculation issues.
    try:
        # This tensor is huge and may not be allocatable on all systems.
        x = torch.randn(1, 10**8, device='cuda', dtype=torch.float32)
    except RuntimeError:
        pytest.skip("Skipping test_no_kernel_launch_error_checking due to memory constraints.")
    
    # Launch the kernel.
    result = cuda_module.forward(x, 1)
    # Force synchronization to catch any latent CUDA errors.
    torch.cuda.synchronize()
    # We compute the expected result using torch.cumsum.
    expected = torch.cumsum(x, dim=1)
    # Because there’s no error checking in the kernel, if the kernel executed out-of-bounds or had an error,
    # the result may silently be wrong. Thus, we assert that the result does NOT match the expected output.
    # (If the kernel were robust, the results would match and the assertion would fail.)
    assert not torch.allclose(result, expected, atol=1e-5), (
        "Kernel unexpectedly produced correct output despite lack of post-launch error checking. "
        "This test is intended to trigger silent CUDA errors."
    )
