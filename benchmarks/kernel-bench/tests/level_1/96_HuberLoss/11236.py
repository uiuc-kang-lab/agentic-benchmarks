
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility to build/load our CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="smooth_l1_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Trigger the issue with non-float32 input (e.g. float64)
def test_non_float32_dtype():
    # Create double precision tensors on CUDA.
    predictions = torch.randn(128, 4096, device="cuda", dtype=torch.float64)
    targets = torch.randn(128, 4096, device="cuda", dtype=torch.float64)
    module = build_kernel()
    # The kernel expects float32.
    result = module.forward(predictions, targets)
    # Reference loss computed in float64 with PyTorch (smooth_l1_loss casts inputs appropriately).
    reference = torch.nn.functional.smooth_l1_loss(predictions.float(), targets.float())
    # Since the kernel computed the loss using double memory accessed as float,
    # the result will be incorrect (and most likely far off from reference).
    assert not torch.allclose(result, reference, atol=1e-5), \
        "Kernel did not fail with float64 inputs as expected."

# Test 2: Trigger the issue with block size assumptions by simulating non-standard block dimensions.
# We simulate this by building a kernel module in which we override the number of elements
# so that the reduction uses fewer than 8 warps. Although the kernel launch uses a fixed block size (256),
# the reduction code unconditionally reads 8 values from shared memory.
# We do this by providing an empty tail (zero elements) for many threads.
def test_shared_memory_reduction_issue():
    # We use a tensor with very few elements so that most threads in the block get no work,
    # but the kernel still launches a block of 256 threads.
    # For a tensor with 16 elements, only the first warp (or two) will be active.
    predictions = torch.randn(16, device="cuda", dtype=torch.float32)
    targets = torch.randn(16, device="cuda", dtype=torch.float32)
    module = build_kernel()
    # The kernel will launch with 256 threads nonetheless.
    # Because the shared memory reduction array is hard-coded to have 8 elements,
    # some threads in the first 8 indices (expected to hold valid warp sums) might read uninitialized memory.
    # We cannot predict exactly the erroneous result but it should differ from the correct loss.
    result = module.forward(predictions, targets)
    reference = torch.nn.functional.smooth_l1_loss(predictions, targets)
    # Check that the kernel's output is not equal to the correct result.
    assert not torch.allclose(result, reference, atol=1e-5), \
        "Kernel shared memory reduction did not show error with non-standard active warps."

# Test 3: Trigger division-by-zero issue with an empty tensor.
def test_division_by_zero():
    # Create empty tensors.
    predictions = torch.empty((0,), device="cuda", dtype=torch.float32)
    targets = torch.empty((0,), device="cuda", dtype=torch.float32)
    module = build_kernel()
    # Expect the kernel to result in a runtime error due to division by zero.
    with pytest.raises(RuntimeError):
        # Launch the kernel; the kernel uses n_elements in the denominator.
        res = module.forward(predictions, targets)
        # Force synchronization to catch asynchronous errors.
        torch.cuda.synchronize()
