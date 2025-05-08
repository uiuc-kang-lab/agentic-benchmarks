
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Compile and load the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Helper: execute the custom kernel 'forward' and compare with torch.relu.
def run_forward(module, input_tensor):
    output = module.forward(input_tensor)
    torch.cuda.synchronize()
    return output

# Issue 1: Incorrect index calculation with offset in vectorized loads.
# To trigger this, we create a large tensor so that the streaming branch is used.
@pytest.mark.cuda
def test_incorrect_offset_indexing():
    # Create a tensor with total elements >= STREAM_THRESHOLD (default 1M) and
    # with a size that is not a multiple of VECTOR_SIZE to force tail scenarios.
    # Using a 1D tensor here.
    total_elements = 1048576 + 100  # exceed threshold and add extra tail elements
    x = torch.randn(total_elements, device="cuda", dtype=torch.float32)
    # Make some entries negative to observe the effect of ReLU.
    x[::17] = -torch.abs(x[::17])
    module = build_kernel()
    output = run_forward(module, x)
    expected = torch.relu(x)
    # If incorrect offset indexing, the streaming branch will produce errors.
    assert torch.allclose(output, expected, atol=1e-5), \
        f"Incorrect indexing with offset detected! Max difference: {(output - expected).abs().max()}"

# Issue 2: Tail element processing.
# Create a small tensor with a number of elements that is not a multiple of VECTOR_SIZE.
@pytest.mark.cuda
def test_tail_element_processing():
    # VECTOR_SIZE is 4 in the kernel; choose a size that gives a remainder (e.g. 7)
    size = 7
    x = torch.tensor([ -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0 ], device="cuda", dtype=torch.float32)
    module = build_kernel()
    output = run_forward(module, x)
    expected = torch.relu(x)
    # If tail elements are not processed correctly, some negatives may persist.
    assert torch.allclose(output, expected, atol=1e-5), \
        f"Tail element processing error! Output: {output.cpu().numpy()}, Expected: {expected.cpu().numpy()}"

# Issue 3: Memory alignment problems from wrong vectorized pointer cast.
# Create a non-contiguous tensor (or misaligned view) if possible.
@pytest.mark.cuda
def test_memory_alignment_issue():
    # Create an aligned tensor first then create a slice that may not be aligned.
    # Although PyTorch generally returns aligned tensors, slicing might break contiguity.
    x_full = torch.randn(1024 + 1, device="cuda", dtype=torch.float32)
    # Exclude the first element to force a potential misalignment.
    x = x_full[1:]
    # Introduce negatives in a pattern to see if ReLU is applied correctly.
    x[::13] = -torch.abs(x[::13])
    module = build_kernel()
    output = run_forward(module, x)
    expected = torch.relu(x)
    assert torch.allclose(output, expected, atol=1e-5), \
        f"Memory alignment issue detected! Output differs from expected."

# Issue 4: Potential issue with unqualified use of min/max.
# While this may be caught at compile time in some environments, we trigger its runtime effect by
# building and running with an input that stresses the branch using min/max.
@pytest.mark.cuda
def test_min_max_function_usage():
    # Use a tensor that forces the kernel to hit the non-vectorized branch (for example, a tensor of size 3)
    x = torch.tensor([-3.0, 0.0, 3.0], device="cuda", dtype=torch.float32)
    module = build_kernel()
    output = run_forward(module, x)
    expected = torch.relu(x)
    assert torch.allclose(output, expected, atol=1e-5), \
        f"Min/max function usage issue detected! Output: {output.cpu().numpy()}, Expected: {expected.cpu().numpy()}"
