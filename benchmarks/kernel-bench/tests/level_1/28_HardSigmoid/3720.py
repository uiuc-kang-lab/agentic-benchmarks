
import torch
import pytest
from torch.nn.functional import hardsigmoid
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

@pytest.fixture(scope="module")
def cuda_module():
    # Build and yield the CUDA module.
    return build_kernel()

# Issue 1: Non-contiguous input is not supported
def test_non_contiguous_input(cuda_module):
    # Create a contiguous input tensor and then create a non-contiguous view by transposing.
    batch_size, dim = 16, 16384
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32)
    x_nc = x.t()  # transpose, making it non-contiguous
    # Run the kernel forward. Since our kernel does not check non-contiguity,
    # the result may be incorrect.
    out = cuda_module.forward(x_nc)
    expected = hardsigmoid(x_nc)
    # The outputs are likely to differ if non-contiguity corrupts indexing.
    assert not torch.allclose(out, expected, atol=1e-5), \
        "Test failed to trigger issue with non-contiguous input (unexpected match)."

# Issue 2: Reduction code assumes blockDim.x >= 32
def test_small_input_for_reduction(cuda_module):
    # Create an input with very few elements so that only one block is launched and
    # only a small number of threads (less than 32) perform the computation.
    # This can reveal issues in the warp-level reduction logic.
    x = torch.randn(1, 16, device="cuda", dtype=torch.float32)  # 16 elements only
    out = cuda_module.forward(x)
    expected = hardsigmoid(x)
    # If reduction is wrong then auxiliary reduction (if later used for statistics)
    # could be off; however, the activation computation is fused and might still be computed correctly.
    # We check here whether the activation result deviates from expected,
    # which might happen if the reduction intrinsics disturb shared memory.
    assert not torch.allclose(out, expected, atol=1e-5), \
        "Test failed to trigger reduction issue with blockDim.x < 32 (unexpected match)."

# Issue 3: No error-check after launch of final reduction kernel
def test_missing_error_check_in_reduction(cuda_module):
    # In a proper kernel, any error in the final reduction should be caught.
    # Here we simulate a scenario by passing an input with an unconventional size so that the 
    # iterative final reduction loop runs many iterations. If there is an error in any reduction loop,
    # the kernel error would be ignored.
    x = torch.randn(1, 1025, device="cuda", dtype=torch.float32)  # 1025 elements forces several reduction steps
    out = cuda_module.forward(x)
    expected = hardsigmoid(x)
    # We intentionally expect the error to surface as a wrong activation computation.
    assert not torch.allclose(out, expected, atol=1e-5), \
        "Test failed to trigger missing error checking in final reduction loop (unexpected match)."

# Issue 4: Kernel assumes contiguous 1D layout (arbitrary strides not handled)
def test_arbitrary_strides(cuda_module):
    # Create a tensor with non-standard strides by slicing.
    x_full = torch.randn(32, 1024, device="cuda", dtype=torch.float32)
    x_slice = x_full[::2, ::3]  # This slice is non-contiguous and with unusual strides.
    out = cuda_module.forward(x_slice)
    expected = hardsigmoid(x_slice)
    assert not torch.allclose(out, expected, atol=1e-5), \
        "Test failed to trigger issue with arbitrary strides (unexpected match)."

# Issue 5: Hard-coded warp mask may yield wrong results with non-standard warp widths
def test_hardcoded_warp_mask(cuda_module):
    # Although warp size is typically 32, we simulate a case by using an input tensor whose number of elements
    # forces the reduction logic to use __shfl_down_sync with an assumed 32-lane warp in a context where fewer lanes
    # are active.
    x = torch.randn(1, 48, device="cuda", dtype=torch.float32)  # 48 elements; not a multiple of 32 in one block reduction.
    out = cuda_module.forward(x)
    expected = hardsigmoid(x)
    assert not torch.allclose(out, expected, atol=1e-5), \
        "Test failed to trigger issue with hard-coded warp mask (unexpected match)."
