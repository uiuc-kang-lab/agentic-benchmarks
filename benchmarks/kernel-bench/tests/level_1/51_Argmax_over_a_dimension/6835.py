
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# This function compiles and loads the CUDA extension from kernel.cu.
def build_kernel():
    module = load(
        name="kernel_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: Type mismatch between intermediate int and output int64_t.
# Although the bug might be subtle, we can design a test where the maximum index is near the upper range of typical int values.
# Here we simulate a scenario where many elements are present along the reduction dimension.
# While we cannot really force an index above INT_MAX in a test, we can at least compare the result to torch.argmax.
def test_type_mismatch_int_vs_int64():
    # Create an input tensor with shape [2, 1024, 8] (innerSize > 1) so that the non-coalesced branch is used.
    # The maximum is placed in a known position.
    torch.manual_seed(0)
    x = torch.randn(2, 1024, 8, device="cuda", dtype=torch.float32)
    # Set a known maximum at a specific index in the reduction dimension (dim=1)
    known_index = 789
    x[0, known_index, :] = 1000.0  # force a unique maximum in first outer element
    module = build_kernel()
    indices = module.forward(x, 1)
    # The expected index from torch.argmax should be known_index for the 0th element.
    # The output tensor shape should be [2,8] (since dimension 1 is removed).
    expected = torch.argmax(x, dim=1)
    assert torch.equal(indices, expected), \
        f"Type mismatch issue: kernel output {indices} != expected {expected}"

# Issue 2: Kernel only accepts float32 inputs.
def test_input_tensor_type():
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.float64)  # double type
    module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        _ = module.forward(x, 1)
    assert "Only float32 is supported" in str(excinfo.value), \
        "Kernel did not complain about non-float32 data type."

# Issue 3: Grid dimension limitations.
# We simulate a scenario that would produce a huge number of output blocks. Actual launch may not be possible,
# so we use a (potentially) oversized shape. We then check if a CUDA launch error is raised.
def test_grid_dimension_limit():
    # Choose dimensions that yield a very large total_outputs.
    # For instance, if outerSize * innerSize exceeds a typical grid.x limit (e.g., > 2^16 for many GPUs).
    # We use small dimSize to not consume too much memory, e.g. 2, but huge outerSize and innerSize.
    outer = 20000
    inner = 20000
    dim_size = 2
    shape = (outer, dim_size, inner)
    # This tensor is huge in virtual grid size (but not in memory since dim_size is small).
    # In practice this may trigger a launch configuration error on many GPUs.
    try:
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
    except RuntimeError:
        pytest.skip("Could not allocate tensor for grid dimension test.")
    module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = module.forward(x, 1)

# Issue 4: Lack of error checking after kernel launch.
# We simulate an error by passing an invalid dimension.
def test_missing_error_check_on_launch():
    # Create a valid tensor but pass an invalid dim (e.g., dim out of range) to trigger host-side TORCH_CHECK.
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.float32)
    module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        _ = module.forward(x, 3)  # There are only 3 dims: indices 0,1,2; so dim==3 is invalid.
    assert "Invalid dim" in str(excinfo.value), "Kernel did not raise error on invalid dimension."

# Issue 5: Handling of empty tensors.
def test_empty_tensor():
    # Create an empty tensor along the reduction dimension.
    x = torch.empty(4, 0, 10, device="cuda", dtype=torch.float32)
    module = build_kernel()
    # The expected output is an empty tensor with the reduction dim removed.
    # Depending on implementation, the kernel might crash on zero-dimension.
    try:
        indices = module.forward(x, 1)
    except RuntimeError as e:
        pytest.fail(f"Kernel failed on empty tensor input: {e}")
    expected_shape = list(x.shape)
    del expected_shape[1]
    assert list(indices.shape) == expected_shape, \
        f"Empty tensor handling error: expected shape {expected_shape}, got {list(indices.shape)}"

