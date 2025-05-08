
import pytest
import torch
from torch.utils.cpp_extension import load
from torch.cuda import Device

def build_kernel():
    # Build the extension module from kernel.cu
    cuda_module = load(
        name="cumsum_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(autouse=True)
def skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("Cuda not available")

def test_large_inner_size():
    """
    Issue 1:
    Create an input tensor with inner_size (product of dimensions after dim) exceeding the maximum threads per block.
    For example, choose a shape where the scanned dimension is not the last, and inner_size > 1024.
    Expect the kernel launch to fail due to too many threads per block.
    """
    # Set dim=1 so that: outer_size = shape[0], stride = shape[1], inner_size = product of dims after dim.
    # We choose inner_size > 1024 by having a last dimension > 1024.
    shape = (2, 16, 1025)  # outer_size = 2, stride = 16, inner_size = 1025 (>1024)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # This call should trigger a kernel launch failure due to blockDim.x being too high.
        module.forward(x, 1)
    torch.cuda.synchronize()

def test_wrong_dtype():
    """
    Issue 2:
    Provide a tensor with type float64 to the kernel.
    The kernel internally calls data_ptr<float>(), which is only valid for float32 tensors.
    We expect an error (or mis-computation) due to the type mismatch.
    """
    shape = (128, 4000)
    x = torch.randn(*shape, device="cuda", dtype=torch.float64).contiguous()
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel is not designed for float64. This should raise an error.
        module.forward(x, 1)
    torch.cuda.synchronize()

def test_incorrect_cumsum_dimension():
    """
    Issue 3:
    The kernel assumes the scanned dimension is contiguous with the trailing dimensions.
    Here we perform cumsum along dim=0 on a tensor which is contiguous in row-major order,
    so the memory layout does not form [outer_size, stride, inner_size] as expected.
    We expect the result to deviate from torch.cumsum.
    """
    # Create a tensor of shape (100, 50, 60) and perform cumsum along dim 0.
    shape = (100, 50, 60)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    module = build_kernel()
    output_kernel = module.forward(x, 0)
    output_ref = torch.cumsum(x, dim=0)
    # They are very likely to differ due to incorrect indexing.
    with pytest.raises(AssertionError):
        assert torch.allclose(output_kernel, output_ref, atol=1e-5), \
            "Kernel cumsum along non-contiguous dim did not deviate from torch.cumsum as expected for this test."

def test_non_unrolled_loop_behavior():
    """
    Issue 4:
    Although testing #pragma unroll behavior performance-wise is non-trivial,
    we create a scenario where the runtime stride (the loop iterations) is not a compile-time constant.
    The test verifies that for tensors with a non-constant stride length, the result is still computed,
    but this may hint that the unrolling is not effective.
    (This test is primarily a placeholder to highlight the potential performance issue; correctness is still enforced.)
    """
    # Use a tensor where stride is not known at compile time (choosing an arbitrary non-trivial value).
    shape = (64, 123, 7)  # outer_size = 64, stride = 123, inner_size = 7 (stride is not a compile-time constant)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    module = build_kernel()
    output_kernel = module.forward(x, 1)
    output_ref = torch.cumsum(x, dim=1)
    assert torch.allclose(output_kernel, output_ref, atol=1e-5), \
        "Kernel output deviates from torch.cumsum even though the issue is expected to be performance related."
    torch.cuda.synchronize()
