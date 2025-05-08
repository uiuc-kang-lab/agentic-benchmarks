
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper function to build the CUDA extension.
def build_kernel():
    # Ensure that the path to kernel.cu is correct relative to this file.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_file = os.path.join(cur_dir, "kernel.cu")
    cuda_module = load(
        name="l1_norm_cuda",
        sources=[cuda_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_issue_non_2d_tensor():
    """
    Issue 1: The kernel only supports 2D tensors.
    We create a 3D tensor, expecting an error or incorrect behavior.
    """
    my_kernel = build_kernel()
    # Create a 3D tensor (e.g., [batch, rows, dim]) which is not supported.
    x = torch.randn(4, 16, 32, device="cuda", dtype=torch.float32)
    with pytest.raises(Exception):
        # Expecting the kernel or the host function to complain about wrong tensor dimension.
        _ = my_kernel.forward(x)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_issue_wrong_type():
    """
    Issue 2: The kernel assumes float32 input.
    We pass in a float64 tensor and expect a failure.
    """
    my_kernel = build_kernel()
    # Create a 2D tensor but as float64.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float64)
    with pytest.raises(Exception):
        # The kernel does not support double precision, so we expect an error.
        _ = my_kernel.forward(x)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_issue_non_contiguous():
    """
    Issue 3: The kernel assumes the input tensor is contiguous.
    We create a non-contiguous tensor and expect unexpected behavior or an error.
    """
    my_kernel = build_kernel()
    # Create a contiguous tensor and then transpose to make it non-contiguous,
    # then reshape it back to 2D with unsqueezing (forcing non-contiguity).
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    x = x.transpose(0, 1)  # Now shape is (16384, 16) and non-contiguous.
    x = x.transpose(0, 1)  # Return to original shape; however, it remains non-contiguous.
    assert not x.is_contiguous(), "Tensor is unexpectedly contiguous."
    with pytest.raises(Exception):
        # Our kernel expects a contiguous input, so it can fail if not recontiguous.
        _ = my_kernel.forward(x)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_issue_no_kernel_launch_error_checking():
    """
    Issue 4: No kernel launch error checking.
    We purposely use a configuration that may trigger an illegal memory access.
    For example, setting D to 0.
    """
    my_kernel = build_kernel()
    # Create a tensor with D=0 which may lead to undefined behavior in the kernel.
    x = torch.randn(16, 0, device="cuda", dtype=torch.float32)
    with pytest.raises(Exception):
        _ = my_kernel.forward(x)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_issue_warp_reduction_mask():
    """
    Issue 5: The warp-level reduction uses a fixed mask (0xffffffff) 
    that does not account for partial warps. We simulate this by setting the thread count
    such that only a subset of a warp is active.
    """
    my_kernel = build_kernel()
    # We simulate this by creating a tensor whose second dimension D is smaller than the typical warp size.
    # This forces only a subset of a warp to be active in the reduction.
    x = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    # The kernel will launch with threads = min(D, 1024), thus 16 threads.
    # The fixed mask might be inappropriate.
    # We are expecting either an error or wrong normalization.
    out = my_kernel.forward(x)
    # Compute reference normalization manually.
    ref = x / (torch.sum(torch.abs(x), dim=1, keepdim=True) + 1e-12)
    # Check if output is not close to reference, indicating the issue.
    assert not torch.allclose(out, ref, atol=1e-5), "Kernel unexpectedly handled partial warp reduction correctly."
