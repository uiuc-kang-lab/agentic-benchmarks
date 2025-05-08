
import pytest
import torch
import time
from torch.utils.cpp_extension import load

# Utility to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Trigger the stride / thread indexing issue.
# If a tensor size forces the kernel to perform more than one loop iteration,
# an incorrect stride (due to including blockDim.y) might lead to errors.
@pytest.mark.cuda
def test_stride_indexing_issue():
    kernel_mod = build_kernel()

    # Choose a tensor size such that the computed grid-stride loop would iterate more than once.
    # With threads=256 and blocks computed as ceil(size/256), the kernel may loop over indices.
    # Here, we pick a size that is not a multiple of (256*number_of_blocks) when blockDim.y is (implicitly) 1.
    size = 10000  # any size that forces at least 2 iterations if stride were miscomputed for 2D blocks
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    
    # Compute ref sigmoid using PyTorch’s native function.
    ref = torch.sigmoid(x)
    out = kernel_mod.forward(x)
    
    # If the stride or indexing is off, then the output will differ from the reference.
    assert torch.allclose(out, ref, atol=1e-5), f"Stride/indexing issue detected. Max diff: {(out-ref).abs().max()}"

# Test 2: Trigger precision issue by passing a double tensor.
# The kernel always uses float precision math so double inputs are processed incorrectly.
@pytest.mark.cuda
def test_double_precision_issue():
    kernel_mod = build_kernel()
    
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    ref = torch.sigmoid(x)
    out = kernel_mod.forward(x)
    
    # Because the kernel casts to float and uses expf,
    # the double-precision result will diverge from the correct result.
    # We expect that the results are NOT nearly equal.
    if torch.allclose(out, ref, atol=1e-8):
        pytest.fail("Double precision issue not triggered: kernel output for double tensor is too close to reference.")

# Test 3: Trigger the stream usage issue.
# By its design, the kernel creates and synchronizes its own stream.
# This blocks asynchronous behavior. In a proper asynchronous workflow,
# operations launched on the current stream would execute without immediate synchronization.
@pytest.mark.cuda
def test_stream_sync_issue():
    kernel_mod = build_kernel()
    
    # Record an event on the current default stream.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Enqueue a dummy asynchronous op on the default stream.
    dummy = torch.randn(1000, device="cuda")
    dummy = dummy * 2  # arbitrary computation
    start_event.record()
    
    # Now call the kernel forward; because it creates its own stream and waits for it,
    # the work on the default stream will not overlap.
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    _ = kernel_mod.forward(x)
    
    end_event.record()
    torch.cuda.synchronize()
    elapsed = start_event.elapsed_time(end_event)
    
    # If the kernel had used the current stream, the elapsed time would be very short.
    # Here, we expect an inflated elapsed time reflecting the forced synchronization.
    # (This test may be fragile; adjust the threshold as needed.)
    if elapsed < 0.5:
        pytest.fail(f"Stream sync issue not triggered: elapsed time {elapsed}ms is unexpectedly short.")
    
# Test 4: Trigger error checking issue by passing an input on the wrong device.
# Because the kernel does not check for input device, a CPU tensor may lead to a segmentation fault
# or undefined behavior. Here, we catch the error.
def test_no_error_checking_issue():
    kernel_mod = build_kernel()
    
    x_cpu = torch.randn(1024, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # This should error out because the kernel is launched on CUDA,
        # but x_cpu is on the CPU.
        _ = kernel_mod.forward(x_cpu)

# Test 5: (Compilation issue for unqualified min)
# This is a compile-time error; however, if the module was built, then the build system
# probably resolved it (or it was not caught). We force a check: try to call forward,
# and if by any chance the kernel misbehaves because of min usage, detect it.
@pytest.mark.cuda
def test_min_function_usage():
    kernel_mod = build_kernel()
    
    # Use an input tensor small enough to not trigger any looping logic.
    x = torch.randn(256, device="cuda", dtype=torch.float32)
    ref = torch.sigmoid(x)
    out = kernel_mod.forward(x)
    
    # If the underlying 'min' was mis-resolved, the kernel might produce an error or wrong output.
    assert torch.allclose(out, ref, atol=1e-5), "Potential issue with min() function resolution detected."

