
import torch
import pytest
from torch.utils.cpp_extension import load

# Function to build the CUDA extension (assumes kernel.cu is in current directory)
def build_kernel():
    cuda_module = load(
        name="test_log_softmax",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger the ambiguous "max" reduction issue.
# We use an input with extreme values so that the correct maximum is obvious.
# If the reduction uses an incorrect max, the output will differ significantly from torch.log_softmax.
def test_max_reduction_bug():
    cuda_kernel = build_kernel()
    # Create an input tensor where one element is much larger than the others.
    # For example, in each row, the maximum should be 100.0.
    batch_size = 4
    dim = 128
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32)
    x[:, 0] = 100.0  # Ensure that the maximum is at index 0 for every row.
    
    # Get kernel result
    y_kernel = cuda_kernel.forward(x, 1)
    # Get reference result via PyTorch
    y_ref = torch.log_softmax(x, dim=1)
    
    # Due to a potential bug in the max reduction, the values might differ significantly.
    max_diff = (y_kernel - y_ref).abs().max()
    assert max_diff < 1e-3, (
        f"Test for max reduction bug failed: maximum difference {max_diff} "
        "exceeds tolerance. Likely issue with the reduction using an unqualified 'max' function."
    )

# Test 2: Trigger the unused warp tile issue.
# Although this does not produce incorrect numerical results in a trivial test, 
# we choose an input with a softmax-dimension not a multiple of 32 to highlight that
# the implementation does not take advantage of warp-level optimizations.
def test_unused_warp_tile():
    cuda_kernel = build_kernel()
    # Use a dimension that is not a multiple of 32.
    batch_size = 4
    dim = 50  # not a multiple of 32
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32)
    
    y_kernel = cuda_kernel.forward(x, 1)
    y_ref = torch.log_softmax(x, dim=1)
    
    max_diff = (y_kernel - y_ref).abs().max()
    assert max_diff < 1e-3, (
        f"Test for unused warp tile failed: maximum difference {max_diff} "
        "exceeds tolerance. Possibly the warp-level reduction was intended but not used."
    )

# Test 3: Trigger the issue suggested by the misleading 'double buffering' comment.
# We use a multi-dimensional input (more than 2D) where the softmax is applied along a non-last dimension.
# The kernel internally permutes to make the softmax dimension last.
def test_double_buffering_comment():
    cuda_kernel = build_kernel()
    # Create a 3D input tensor with random values.
    # We choose dimension 1 (second dimension) as the softmax dimension.
    batch_size, channels, features = 4, 10, 20
    x = torch.randn(batch_size, channels, features, device="cuda", dtype=torch.float32)
    
    # Because the kernel is written to work with the softmax dimension as the last,
    # it internally permutes the input. If the commentary or implementation about buffering
    # is misleading, this may cause subtle numerical differences.
    y_kernel = cuda_kernel.forward(x, 1)
    y_ref = torch.log_softmax(x, dim=1)
    
    max_diff = (y_kernel - y_ref).abs().max()
    assert max_diff < 1e-3, (
        f"Test for double buffering comment issue failed: maximum difference {max_diff} "
        "exceeds tolerance. This might be related to the shared memory usage not matching the comment."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
