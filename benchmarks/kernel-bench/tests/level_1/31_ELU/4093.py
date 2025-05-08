
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import threading
import time

def build_kernel():
    # Build/load the extension from kernel.cu
    cuda_module = load(
        name="elu_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope='module')
def kernel_module():
    return build_kernel()

# Test case 1: Trigger the input data type issue by using a float64 tensor.
def test_input_data_type_issue(kernel_module):
    # Create a CUDA tensor with double precision.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # Compute reference result using PyTorch ELU (will work with any dtype)
    ref = F.elu(x, alpha=1.0)
    
    # Calling our kernel (which will interpret the memory as float32) should yield wrong results.
    # We do not expect an exception here but an incorrect result.
    # Note: This error may be silent so we check that the result deviates significantly from the reference.
    out = kernel_module.forward(x, 1.0)
    # Convert output to float64 for comparison (it was computed as if the input were float32)
    out = out.to(torch.float64)
    
    # Check that the outputs are not close.
    assert not torch.allclose(out, ref, atol=1e-3), "Kernel incorrectly accepted non-float32 tensor without error."

# Test case 2: Trigger the concurrent invocation issue by using different alpha values on different streams.
def test_concurrent_alpha_issue(kernel_module):
    # Prepare an input tensor and two different alpha values.
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    alpha1 = 1.0
    alpha2 = 2.0

    # Function to call the kernel in a dedicated CUDA stream.
    def run_kernel(alpha, out_container, idx):
        stream = torch.cuda.Stream()
        # Enqueue operations in a different stream.
        with torch.cuda.stream(stream):
            # Launch the kernel asynchronously.
            result = kernel_module.forward(x, alpha)
            # Make sure the work completes in this stream.
            stream.synchronize()
            out_container[idx] = result

    # Container for storing results from two threads.
    results = [None, None]

    # Run two threads concurrently.
    t1 = threading.Thread(target=run_kernel, args=(alpha1, results, 0))
    t2 = threading.Thread(target=run_kernel, args=(alpha2, results, 1))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Reference calculation using PyTorch for each alpha.
    ref1 = F.elu(x, alpha=alpha1)
    ref2 = F.elu(x, alpha=alpha2)

    # Because of the race on constant memory, it is possible that one or both outputs do not match their reference.
    # We check that at least one of the computations is incorrect.
    err1 = (results[0] - ref1).abs().max().item()
    err2 = (results[1] - ref2).abs().max().item()
    # We set a threshold which should be very small if the computation were correct.
    threshold = 1e-3
    assert err1 > threshold or err2 > threshold, \
      "Kernel executed concurrently with different alpha values produced results that are too similar to reference. " \
      "This indicates that the global constant memory update may not be thread-safe."

# Test case 3: Trigger the non-contiguous input issue.
def test_non_contiguous_input(kernel_module):
    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x = torch.randn(128, 128, device="cuda", dtype=torch.float32)
    x_non_contiguous = x.t()  # Transpose makes it non-contiguous.
    with pytest.raises(RuntimeError, match="must be contiguous"):
        # The CHECK_INPUT macro should trigger a RuntimeError.
        kernel_module.forward(x_non_contiguous, 1.0)
