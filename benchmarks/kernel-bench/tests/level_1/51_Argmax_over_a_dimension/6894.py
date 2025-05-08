
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build and load the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_argmax_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test Case 1: Trigger error by passing a non-float32 tensor (e.g. half-precision)
def test_input_type_not_float32():
    my_module = build_kernel()
    # Create a half-precision tensor. The kernel checks for float32 and should trigger an error.
    x = torch.randn(4, 5, 6, device="cuda", dtype=torch.half)
    with pytest.raises(RuntimeError, match="Only float32 is supported."):
        # The kernel launch should trigger the TORCH_CHECK error.
        my_module.forward(x, 1)

# Test Case 2: Trigger reduction boundary issue by simulating a kernel launch with fewer threads.
# To simulate a situation where blockDim.x < 32, we can force a tensor with innerSize and outerSize = 1 
# and then modify the kernel launch configuration via monkeypatching the ctypes attribute.
# Since the kernel launch parameters are hardcoded in the C++ code, we simulate the condition 
# by providing an input tensor where the reduction loop itself is stressed.
def test_launch_with_small_reduction_range(monkeypatch):
    my_module = build_kernel()
    # Create an input tensor with shape that forces a small 'dimSize'
    # Here, dimSize will be small relative to the fixed 256 threads so that many threads
    # do no work. Although the kernel does not adjust blockDim.x, the reduction stage will 
    # read shared memory entries beyond the active thread range.
    x = torch.randn(1, 2, 1, device="cuda", dtype=torch.float32)
    # The expected behavior is undefined, but we want to ensure that the kernel raises an error.
    # We check whether the kernel call leads to a CUDA error reported by torch.
    with pytest.raises(RuntimeError):
        my_module.forward(x, 1)

# Test Case 3: Trigger error checking absence by providing an invalid dimension.
def test_invalid_dimension():
    my_module = build_kernel()
    x = torch.randn(4, 5, 6, device="cuda", dtype=torch.float32)
    # Use an invalid dimension value that is out of range.
    with pytest.raises(RuntimeError, match="Invalid dim"):
        my_module.forward(x, 3)

# Test Case 4: Trigger grid dimension overflow for very large inputs.
# This test provides an input with extremely large outer and inner sizes.
# Note: In a real-world scenario, the extremely large tensor might not be allocatable;
# therefore, we simulate the error by passing dimensions that will lead to a grid size
# that exceeds CUDA’s limits.
def test_grid_dimension_overflow():
    my_module = build_kernel()
    # Simulate a case where outerSize * innerSize becomes enormous.
    # Instead of really allocating a huge tensor (which would exhaust memory),
    # we use torch.empty with the appropriate sizes and then call forward.
    # We use a non-standard shape that, if unguarded, would set grid dimension too high.
    # For instance, suppose the reduction dimension is very small while the others are huge.
    try:
        # Intentionally small dim size but huge outer and inner
        shape = (2**16, 2**16, 2)  # outerSize and innerSize become huge (2^16 each)
        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    except RuntimeError:
        pytest.skip("Skipping test: Unable to allocate tensor for grid overflow test.")

    with pytest.raises(RuntimeError):
        my_module.forward(x, 2)
