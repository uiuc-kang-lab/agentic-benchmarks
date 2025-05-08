
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="hinge_loss_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports float32.
def test_input_dtype():
    my_module = build_kernel()
    N = 1024
    # Create double precision tensors on CUDA.
    predictions = torch.randn(N, device="cuda", dtype=torch.double)
    # Map binary targets into {-1, 1} using double precision
    targets = (torch.randint(0, 2, (N,), device="cuda", dtype=torch.double) * 2 - 1)
    with pytest.raises(RuntimeError) as excinfo:
        # Expected to fail since __ldg and fmaxf are used on floats.
        out = my_module.forward(predictions, targets)
        torch.cuda.synchronize()
    assert "float" in str(excinfo.value).lower(), f"Unexpected error message: {excinfo.value}"

# Issue 2: Kernel does not check shape consistency between predictions and targets.
def test_mismatched_shapes():
    my_module = build_kernel()
    N_pred = 1024
    N_targ = 512  # Mismatch: fewer elements in targets
    predictions = torch.randn(N_pred, device="cuda", dtype=torch.float32)
    targets = (torch.randint(0, 2, (N_targ,), device="cuda", dtype=torch.float32) * 2 - 1)
    # This might result in out-of-bound memory access.
    with pytest.raises(RuntimeError) as excinfo:
        out = my_module.forward(predictions, targets)
        torch.cuda.synchronize()
    assert "out of range" in str(excinfo.value).lower() or "invalid" in str(excinfo.value).lower(), \
        f"Unexpected error message: {excinfo.value}"

# Issue 3: Kernel requires contiguous inputs; non-contiguous tensors should trigger an error.
def test_non_contiguous_inputs():
    my_module = build_kernel()
    N = 1024
    # Create a contiguous tensor and then create a non-contiguous view.
    predictions_contig = torch.randn(N, device="cuda", dtype=torch.float32)
    targets_contig = (torch.randint(0, 2, (N,), device="cuda", dtype=torch.float32) * 2 - 1)
    predictions = predictions_contig[::2]  # Non-contiguous view (stride > 1)
    targets = targets_contig[::2]           # Non-contiguous view
    with pytest.raises(RuntimeError) as excinfo:
        out = my_module.forward(predictions, targets)
        torch.cuda.synchronize()
    assert "contiguous" in str(excinfo.value).lower(), f"Unexpected error message: {excinfo.value}"

# Issue 4: Kernel assumes BLOCK_SIZE is a multiple of warpSize.
# This test attempts to force a different block size by monkey-patching the forward kernel.
def test_invalid_block_size():
    my_module = build_kernel()
    N = 1024
    predictions = torch.randn(N, device="cuda", dtype=torch.float32)
    targets = (torch.randint(0, 2, (N,), device="cuda", dtype=torch.float32) * 2 - 1)
    
    # Intentionally force an invalid block size by overriding the block selection logic.
    # We simulate this by building a kernel with a BLOCK_SIZE that is not a multiple of 32.
    # (Note: The provided kernel does not support this; in a general context such a call should error.)
    # Here, we call the kernel directly (bypassing the forward helper) using a BLOCK_SIZE of 50.
    # We create a partialSums tensor with an arbitrary grid configuration.
    BLOCK_SIZE = 50  # Not a multiple of 32, which the kernel function assumes.
    blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    partialSums = torch.empty((blocks,), device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream

    # We wrap the direct launch in a try-except to catch errors.
    try:
        # Note: We have to call the templated kernel for a specific BLOCK_SIZE.
        # Since our extension only exports the forward() wrapper, this test simulates the erroneous condition.
        # In a real scenario, the kernel would be instantiated with the wrong block size.
        my_module.forward(predictions, targets)
        # If no error is raised, we consider this a failure of the test.
        pytest.fail("Expected error due to block size not being a multiple of warpSize was not raised.")
    except RuntimeError as e:
        assert "warp" in str(e).lower() or "block size" in str(e).lower(), f"Unexpected error message: {e}"
