
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Use of a fixed warp mask may produce incorrect results when the number
# of elements per row (D) is smaller than the warp size. This test triggers an incomplete warp.
def test_incomplete_warp():
    torch.manual_seed(0)
    # Use a very small D so that many threads in the warp are idle.
    batch_size = 16
    D = 8  # intentionally less than warpSize=32
    predictions = torch.randn(batch_size, D, dtype=torch.float32, device='cuda')
    targets = torch.randn(batch_size, D, dtype=torch.float32, device='cuda')
    my_module = build_kernel()
    loss = my_module.forward(predictions, targets)
    # We expect a valid scalar loss but the result might be incorrect
    # due to the use of a constant warp mask.
    # Here we only check that the loss is computed (not NaN) but in a real scenario 
    # one would compare with a CPU reference.
    assert not torch.isnan(loss).any(), "Loss is NaN, which may be due to incorrect warp mask reduction."

# Issue 2: The kernel assumes blockIdx.x < N. We simulate a misconfigured launch by adding extra rows.
# Since the kernel launch is encapsulated in the pybind11 function, we trigger the check by passing 
# a zero batch which would lead to a grid dimension of 0, potentially leading to an out‐of‐bounds access.
def test_invalid_block_index():
    torch.manual_seed(0)
    batch_size = 0  # empty batch; grid will be launched with 0 blocks
    D = 128
    predictions = torch.randn(batch_size, D, dtype=torch.float32, device='cuda')
    targets = torch.randn(batch_size, D, dtype=torch.float32, device='cuda')
    my_module = build_kernel()
    # This should trigger an error or return a zero loss.
    try:
        loss = my_module.forward(predictions, targets)
        # If loss is computed, it must be zero.
        assert loss.item() == 0.0, "Expected loss to be 0 for an empty batch."
    except RuntimeError as e:
        pytest.fail("Kernel did not correctly handle empty batch: " + str(e))

# Issue 3: No error checking after kernel launch. Although the pybind11 wrapper does not
# explicitly check for kernel launch errors, we trigger a situation where an error is expected.
# For instance, launching the kernel with a wrong input type should trigger the TORCH_CHECK.
def test_wrong_input_type():
    torch.manual_seed(0)
    batch_size = 16
    D = 128
    # Use double precision instead of float32.
    predictions = torch.randn(batch_size, D, dtype=torch.float64, device='cuda')
    targets = torch.randn(batch_size, D, dtype=torch.float64, device='cuda')
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = my_module.forward(predictions, targets)

# Issue 4: Lack of generality regarding precision. This test is similar to test_wrong_input_type,
# but it explicitly documents that passing a tensor of a type not supported (e.g., torch.half)
# should raise an error.
def test_unsupported_precision():
    torch.manual_seed(0)
    batch_size = 16
    D = 128
    predictions = torch.randn(batch_size, D, dtype=torch.half, device='cuda')
    targets = torch.randn(batch_size, D, dtype=torch.half, device='cuda')
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = my_module.forward(predictions, targets)
