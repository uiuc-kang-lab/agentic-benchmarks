
import os
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility to build the kernel module.
# We also allow for injecting custom definitions via extra_cuda_cflags.
def build_kernel(extra_cuda_cflags=None):
    flags = ["-O3", "--use_fast_math"]
    if extra_cuda_cflags is not None:
        flags.extend(extra_cuda_cflags)
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=flags,
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# ----- Issue 1: Incorrect KL divergence formula -----
def test_incorrect_formula():
    # Prepare a simple input where the correct kl_div can be computed
    # using PyTorch and compared against our custom kernel.
    # Create two small probability distributions.
    torch.manual_seed(123)
    batch_size, dim = 4, 8
    # The model code expects probabilities so we use softmax along dim=-1.
    predictions = torch.randn(batch_size, dim, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch_size, dim, device="cuda").softmax(dim=-1)
    
    # Use PyTorch's kl_div function for expected outcome.
    # Note: torch.nn.functional.kl_div expects the input to be log-probabilities.
    expected = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    
    module = build_kernel()
    output = module.forward(torch.log(predictions), targets)
    torch.cuda.synchronize()
    # Because the kernel uses a wrong formula, the result will differ.
    assert not torch.allclose(output, expected, atol=1e-5), \
        f"Test failed: The kernel computed value matches the expected value, but it is using the wrong formula. Kernel output: {output}, Expected: {expected}"

# ----- Issue 2: Incorrect divisor in batchmean reduction -----
def test_incorrect_divisor():
    # Create input with nontrivial spatial dimensions so that the total element count n != batch size.
    torch.manual_seed(456)
    batch_size, dim = 8, 128  # total elements = 8*128 = 1024 per tensor
    predictions = torch.randn(batch_size, dim, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch_size, dim, device="cuda").softmax(dim=-1)
    
    # Expected output using PyTorch's kl_div expects division by batch_size
    expected = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    
    module = build_kernel()
    output = module.forward(torch.log(predictions), targets)
    torch.cuda.synchronize()
    
    # Because the kernel divides by the total number of elements instead of batch size,
    # the scale of the output will be different.
    if torch.allclose(output, expected, atol=1e-5):
        raise AssertionError(f"Test failed: Kernel reduction appears correct. Kernel output: {output}, Expected: {expected}")
    else:
        # A warning message to indicate that the reduction error is detected.
        print("Reduction divisor error detected: kernel output differs from expected batchmean reduction.")

# ----- Issue 3: Fixed shared memory allocation not general for large blockDim -----
def test_shared_memory_limit():
    # We simulate the situation where blockDim.x/32 > 32.
    # To do so, we force the kernel to use a larger thread count per block.
    # Here we inject a macro definition for THREADS_PER_BLOCK and modify the launch configuration.
    # (Assuming kernel.cu is modified to use the macro THREADS_PER_BLOCK instead of hard-coded 512.)
    extra_flags = ["-DTHREADS_PER_BLOCK=1056"]
    module = build_kernel(extra_cuda_cflags=extra_flags)
    
    # Create an input that forces many threads. The size is chosen so that
    # the kernel launch uses our forced THREADS_PER_BLOCK value.
    n_elements = 1056 * 10  # arbitrary
    predictions = torch.randn(n_elements, device="cuda").softmax(dim=0)
    targets = torch.randn(n_elements, device="cuda").softmax(dim=0)
    
    try:
        output = module.forward(torch.log(predictions), targets)
        torch.cuda.synchronize()
        # Since the shared memory is not allocated for more than 32 warps, behavior is undefined.
        # If no error is raised, we still flag that the kernel is not general.
        raise AssertionError("Test failed: Kernel did not crash or produce an error when using blockDim with more than 32 warps.")
    except RuntimeError as e:
        # Expecting a CUDA error because of shared memory overrun.
        print("Shared memory limit issue detected (as expected):", str(e))

# ----- Issue 4: No input type and contiguity checks -----
def test_input_tensor_type():
    # Pass input tensors with double precision rather than float32.
    predictions = torch.randn(128, 4096, device="cuda", dtype=torch.float64).softmax(dim=-1)
    targets = torch.randn(128, 4096, device="cuda", dtype=torch.float64).softmax(dim=-1)
    
    module = build_kernel()
    try:
        output = module.forward(torch.log(predictions), targets)
        torch.cuda.synchronize()
        raise AssertionError("Test failed: Kernel accepted double precision tensors; it should require float32 inputs.")
    except RuntimeError as e:
        # Expecting a runtime error due to type mismatch because the kernel works only with float.
        print("Input tensor type issue detected (as expected):", str(e))

