
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel(extra_cuda_cflags=None):
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        verbose=True,
        extra_cuda_cflags=extra_cuda_cflags if extra_cuda_cflags else ["-O3", "--use_fast_math"],
    )
    return cuda_module

# Issue 1: Incorrect reduction factor.
# Using a batch size > 1 the expected reduction ("batchmean") divides by the batch size,
# but our kernel divides by the total number of elements.
def test_incorrect_reduction_factor():
    # Create an input where each row is a valid softmax distribution.
    # Use a batch size that is different from the length of the distribution.
    batch_size = 128
    dist_len = 4096
    # random predictions and targets such that both are valid probability distributions
    predictions = torch.randn(batch_size, dist_len, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch_size, dist_len, device="cuda").softmax(dim=-1)
    # Build the module
    module = build_kernel()
    # Compute KL divergence using our CUDA kernel
    loss_cuda = module.forward(torch.log(predictions), targets)
    # Compute KL divergence via torch.nn.functional.kl_div (which uses batchmean reduction)
    loss_torch = F.kl_div(torch.log(predictions), targets, reduction='batchmean')
    
    # Because the kernel divides by n (number of elements), the result will be scaled differently.
    # We expect the relative difference to be significant.
    rel_diff = abs(loss_cuda.item() - loss_torch.item()) / (abs(loss_torch.item()) + 1e-8)
    assert rel_diff > 0.1, f"Reduction factor issue not detected: rel_diff = {rel_diff}"

# Issue 2: Incorrect KL divergence formula.
# When predictions equal targets (so that the true KL divergence should be 0), the correct implementation
# would return 0. However, the kernel produces nonzero output because it computes an extra term.
def test_incorrect_formula():
    batch_size = 16
    dist_len = 64
    # Create distributions that are identical: uniform over the distribution
    uniform = torch.full((batch_size, dist_len), 1.0/dist_len, device="cuda")
    predictions = uniform
    targets = uniform
    module = build_kernel()
    loss_cuda = module.forward(torch.log(predictions), targets)
    loss_torch = F.kl_div(torch.log(predictions), targets, reduction='batchmean')
    # Expected loss should be 0 (or numerically very close) when both distributions are identical.
    # The CUDA kernel result will be nonzero if the formula is wrong.
    assert abs(loss_torch.item()) < 1e-6, f"Reference kl_div not near zero: {loss_torch.item()}"
    assert abs(loss_cuda.item()) > 1e-3, f"Kernel formula issue not detected (loss_cuda={loss_cuda.item()})"

# Issue 3: Implicit assumption of float32 input.
# Passing a tensor of type double (float64) should cause issues (usually a crash or wrong output) 
# because the kernel fetches data as float.
def test_input_tensor_type():
    batch_size = 32
    dist_len = 128
    predictions = torch.randn(batch_size, dist_len, device="cuda", dtype=torch.double).softmax(dim=-1)
    targets = torch.randn(batch_size, dist_len, device="cuda", dtype=torch.double).softmax(dim=-1)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Attempt to call the CUDA kernel with non-float32 tensors.
        # This should raise an error or produce a runtime exception.
        _ = module.forward(torch.log(predictions), targets)

# Issue 4: Implicit assumption that blockDim.x is a multiple of 32.
# We simulate this scenario by forcing a kernel build where the threads-per-block is overridden.
# In our kernel the threads_per_block is hardcoded to 128.
# We mimic a "non-multiple" case by providing a custom extra CUDA flag to define a macro that
# overrides the number of threads per block, then cause the reduction logic to fail.
def test_incorrect_block_configuration():
    # We override the threads per block to a value that is not a multiple of 32.
    # This test assumes that the kernel code is modified to use a macro for threads_per_block.
    # For demonstration, we use -DTHREADS_PER_BLOCK=50 to force an improper configuration.
    extra_flags = ["-O3", "--use_fast_math", "-DTHREADS_PER_BLOCK=50"]
    module = build_kernel(extra_cuda_cflags=extra_flags)
    
    # We create a small input.
    batch_size = 8
    dist_len = 100  # total elements = 800, not necessarily a multiple of 50 or 32
    predictions = torch.randn(batch_size, dist_len, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch_size, dist_len, device="cuda").softmax(dim=-1)
    
    loss_cuda = module.forward(torch.log(predictions), targets)
    loss_torch = F.kl_div(torch.log(predictions), targets, reduction='batchmean')
    
    # Because the kernel's warp reduction may operate incorrectly when blockDim.x is not a multiple of 32,
    # the result is expected to be quite incorrect.
    rel_diff = abs(loss_cuda.item() - loss_torch.item()) / (abs(loss_torch.item()) + 1e-8)
    assert rel_diff > 0.1, f"Block configuration issue not detected: rel_diff = {rel_diff}"
