
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Rebuild the extension each time so that tests trigger any potential launch/config issues.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Trigger type mismatch by passing a tensor of type double.
def test_non_float_input():
    my_module = build_kernel()
    # Create a double tensor
    x = torch.randn(16, 256, 256, dtype=torch.double, device='cuda')
    dim = 1
    with pytest.raises(RuntimeError, match="Only float32 supported"):
        # The kernel forward should check and raise error for non float32 input.
        my_module.forward(x, dim)

# Test 2: Trigger invalid dimension usage by passing an out‐of‐range dimension.
def test_invalid_dim():
    my_module = build_kernel()
    x = torch.randn(16, 256, 256, dtype=torch.float32, device='cuda')
    invalid_dim = 5  # Given tensor has 3 dimensions.
    with pytest.raises(RuntimeError, match="Invalid dim"):
        my_module.forward(x, invalid_dim)

# Test 3: Trigger exceeding grid dimension limits.
# This test attempts to create a tensor with a huge grid (outerSize * innerSize)
# and catches the CUDA launch error.
def test_exceed_grid_limits():
    my_module = build_kernel()
    # Instead of truly enormous sizes (which might crash the system),
    # we simulate the condition by setting one of the dimensions to an extremely large value.
    # Note: This test may require a system where kernel launch failures can be caught.
    # We use a try/except to capture the CUDA launch error.
    try:
        # Construct a tensor with a huge outer*inner product.
        # Here, we choose dimensions that are unrealistic.
        # e.g., outerSize = 2**16, innerSize = 2**16, and dimSize = 2.
        # The total grid size would be 2**32, which should exceed limits.
        outer = 2**16
        inner = 2**16
        dimSize = 2
        # Build a tensor of shape [outer, dimSize, inner]
        x = torch.randn(outer, dimSize, inner, dtype=torch.float32, device='cuda')
        # This should cause a launch configuration error due to the large grid.
        my_module.forward(x, 1)
    except RuntimeError as e:
        assert "grid" in str(e).lower() or "launch" in str(e).lower(), f"Unexpected error message: {e}"
    else:
        pytest.skip("The current GPU supports the simulated grid dimension; test skipped.")

# Test 4: Simulation of non-power-of-two blockDim reduction issues.
# Although blockDim is fixed in the host code, we simulate a scenario that could expose reduction issues.
# We create an input tensor with a very small dimSize to possibly make many threads in the block do no work.
def test_reduction_edge_case():
    my_module = build_kernel()
    # Create a tensor with a very small reduction dimension.
    # For example, let dimSize (the size along the argmax dimension) be 1.
    x = torch.randn(16, 1, 256, dtype=torch.float32, device='cuda')
    # The expected behavior is that argmax always returns index 0 along that dimension.
    indices = my_module.forward(x, 1)
    expected = torch.zeros([16, 256], dtype=torch.long, device='cuda')
    assert torch.equal(indices, expected), f"Reduction failed: got {indices}, expected {expected}"
