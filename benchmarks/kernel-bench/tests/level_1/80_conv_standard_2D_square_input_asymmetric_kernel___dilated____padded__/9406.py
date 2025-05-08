
import os
import torch
import time
import pytest
from torch.utils.cpp_extension import load

# Utility: build the module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 Test: Check that the kernel file does not implement tiling.
# We expect it to be “tiled” (i.e. use shared memory) but since no shared memory is allocated,
# we treat the absence of __shared__ keyword as an issue.
def test_missing_tiling():
    kernel_file = "kernel.cu"
    assert os.path.isfile(kernel_file), f"{kernel_file} does not exist!"
    with open(kernel_file, "r") as f:
        contents = f.read()
    # If the file contained a tiling implementation, we might expect to see shared memory declarations.
    assert "__shared__" in contents, "Kernel is labeled as tiled but does not use shared memory (tiling)!"

# Issue 2 Test: Attempt to trigger a grid dimension limit violation.
# We query the CUDA device properties and then set out_channels to exceed maxGridSize[2].
def test_exceed_grid_dim_z():
    # Get device properties
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    # gridDim.z maximum. This number can vary; on many devices it is limited.
    max_grid_z = props.maxGridSize[2]
    # We set out_channels to exceed the maximum gridDim.z limit.
    # (Note: if the device has a very high gridDim.z limit this test might be inconclusive.)
    out_channels = max_grid_z + 10  
    in_channels = 3
    kernel_h, kernel_w = 3, 5
    batch_size = 1
    input_height, input_width = 32, 32

    # Create random input, weight and a dummy bias (set bias to None so kernel uses no bias value)
    x = torch.randn(batch_size, in_channels, input_height, input_width, device=device, dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device=device, dtype=torch.float32)
    # Note: We pass an empty tensor as bias if needed; here we pass None to simulate no bias.
    cuda_module = build_kernel()

    # Expect a CUDA error because gridDim.z is exceeded.
    with pytest.raises(RuntimeError):
        # The call should trigger a launch error due to grid dimension overflow.
        _ = cuda_module.forward(x, weight, None, 1, (1,2), (2,1))
        torch.cuda.synchronize()

# Issue 3 Test: Check for serial batch processing.
# A properly parallelized kernel would process all batch elements concurrently.
# Since the kernel loops over the batch dimension inside each thread, running multiple batches 
# will cause an (approximately) proportional increase in runtime.
#
# We measure the kernel execution time for batch_size=1 and batch_size=4 (keeping all other sizes the same)
# and then check that the relative time increase is near the batch factor (which indicates serial processing).
def test_serial_batch_processing():
    device = torch.device("cuda")
    cuda_module = build_kernel()
    
    in_channels = 3
    out_channels = 16
    kernel_h, kernel_w = 3, 5
    input_height, input_width = 64, 64
    stride = 1
    padding = (1, 2)
    dilation = (2, 1)
    
    def run_kernel(batch):
        x = torch.randn(batch, in_channels, input_height, input_width, device=device, dtype=torch.float32)
        weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out = cuda_module.forward(x, weight, None, stride, padding, dilation)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        return elapsed_ms, out

    # Run with batch size 1 and then size 4
    time1, out1 = run_kernel(1)
    time4, out4 = run_kernel(4)
    
    # Compute the average time per sample in each case.
    avg1 = time1 / 1
    avg4 = time4 / 4
    
    # Because the kernel loops serially over batch indices, we expect avg4 to be noticeably higher.
    # In a well-parallelized kernel, the processing time per sample would be nearly the same.
    # We set an arbitrary threshold: if avg4 is less than 1.5x avg1, then the batch processing might have been parallelized.
    # Since our design forces serial processing, we flag it if we see a significant degradation.
    assert avg4 > 1.5 * avg1, (
        f"Kernel appears to process batches in parallel (avg time per sample: {avg4:.2f}ms vs {avg1:.2f}ms), "
        "but expected serial processing over the batch dimension."
    )
