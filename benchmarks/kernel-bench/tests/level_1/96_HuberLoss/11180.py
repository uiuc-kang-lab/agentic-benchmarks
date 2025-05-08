
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu located in the same directory as this test file.
def build_kernel():
    cuda_module = load(
        name="custom_smooth_l1_loss",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# 1. Test misaligned memory access
def test_misaligned_memory():
    my_module = build_kernel()
    N = 4097  # Ensure size is not a multiple of 4.
    # Create a tensor with an extra element at the beginning so that slicing off the first element may cause misalignment.
    base = torch.randn(128, N + 1, device="cuda", dtype=torch.float32)
    # Slice so that the resulting tensor does not start at the original aligned pointer.
    predictions = base[:, 1:]
    targets = base[:, :-1]
    
    # Compute expected smooth L1 loss using PyTorch's function.
    expected = torch.nn.functional.smooth_l1_loss(predictions, targets)
    # Run the custom CUDA kernel.
    result = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    # The misaligned access may produce an incorrect result.
    assert not torch.allclose(result, expected, atol=1e-5), \
        "Test misaligned_memory: Kernel unexpectedly matched the expected output despite misalignment."

# 2. Test non-multiple-of-4 element count for vectorized load safety
def test_non_multiple_four():
    my_module = build_kernel()
    # Use a tensor whose total number of elements is not divisible by 4.
    # For example, 128 x 4097 is not divisible by 4.
    predictions = torch.randn(128, 4097, device="cuda", dtype=torch.float32)
    targets = torch.randn(128, 4097, device="cuda", dtype=torch.float32)
    
    expected = torch.nn.functional.smooth_l1_loss(predictions, targets)
    result = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    # The vectorized loop may cause invalid reads leading to wrong output.
    assert not torch.allclose(result, expected, atol=1e-5), \
        "Test non_multiple_four: Kernel unexpectedly computed correct results when input size is not a multiple of 4."

# 3. Test warp-level reduction correctness with block dimensions not multiple of warpSize
def test_warp_reduction():
    my_module = build_kernel()
    # Choose a size that forces the kernel to use a number of threads per block that is not a multiple of 32.
    # We simulate this by creating an input where total elements lead to a block launch
    # that is not a multiple of the warp size when processed as float4.
    predictions = torch.randn(130, device="cuda", dtype=torch.float32)
    targets = torch.randn(130, device="cuda", dtype=torch.float32)
    
    expected = torch.nn.functional.smooth_l1_loss(predictions, targets)
    result = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    # If warp-level reduction has an indexing issue, the result will be off.
    assert not torch.allclose(result, expected, atol=1e-5), \
        "Test warp_reduction: Kernel unexpectedly produced correct reduction results."
    
# 4. Test non-thread-safe constant memory update in concurrent launches
def test_constant_memory_concurrency():
    my_module = build_kernel()
    # Create two different pairs of tensors with different sizes
    predictions1 = torch.randn(64, 1024, device="cuda", dtype=torch.float32)
    targets1 = torch.randn(64, 1024, device="cuda", dtype=torch.float32)
    
    predictions2 = torch.randn(128, 2049, device="cuda", dtype=torch.float32)
    targets2 = torch.randn(128, 2049, device="cuda", dtype=torch.float32)
    
    # Launch two kernel calls on different streams concurrently.
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    with torch.cuda.stream(stream1):
        res1 = my_module.forward(predictions1, targets1)
    with torch.cuda.stream(stream2):
        res2 = my_module.forward(predictions2, targets2)
    
    torch.cuda.synchronize()
    
    expected1 = torch.nn.functional.smooth_l1_loss(predictions1, targets1)
    expected2 = torch.nn.functional.smooth_l1_loss(predictions2, targets2)
    
    # Due to concurrent constant memory updates, at least one of the results is expected to be incorrect.
    flag1 = torch.allclose(res1, expected1, atol=1e-5)
    flag2 = torch.allclose(res2, expected2, atol=1e-5)
    
    assert not (flag1 and flag2), \
        "Test constant_memory_concurrency: Both concurrent kernel executions unexpectedly produced correct results."
    
# 5. Test behavior for unsupported data types (e.g., double precision)
def test_double_precision():
    my_module = build_kernel()
    predictions = torch.randn(128, 4096, device="cuda", dtype=torch.float64)
    targets = torch.randn(128, 4096, device="cuda", dtype=torch.float64)
    
    # The kernel expects float32 data; running on double should yield incorrect result.
    expected = torch.nn.functional.smooth_l1_loss(predictions.float(), targets.float())
    # Cast to float64 so that torch tensor passes the contiguity/device checks,
    # but underlying memory will be misinterpreted by the kernel.
    result = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    assert not torch.allclose(result, expected, atol=1e-5), \
        "Test double_precision: Kernel unexpectedly produced correct results for double precision inputs."
