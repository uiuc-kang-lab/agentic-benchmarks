
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 & 2: Incorrect KL divergence formula and wrong scaling factor.
def test_incorrect_kl_div_formula():
    # We create a small batch so that we can compute the expected KL divergence.
    # Using a known case: for a distribution p and target q, torch.nn.functional.kl_div
    # expects input as log(p) and target as q, and computes sum(q * (log(q) - log(p))) / batch_size.
    batch_size = 4
    numel = 8
    # Create valid probability distributions.
    A = torch.rand(batch_size, numel, device="cuda", dtype=torch.float32).softmax(dim=-1)
    B = torch.rand(batch_size, numel, device="cuda", dtype=torch.float32).softmax(dim=-1)
    log_A = torch.log(A)
    
    # Expected computation using PyTorch's built-in function.
    expected = torch.nn.functional.kl_div(log_A, B, reduction="batchmean")
    
    # Run the custom kernel.
    kl_div_module = build_kernel()
    # Custom kernel expects log_predictions and targets; note that it divides by n.
    kernel_out = kl_div_module.forward(log_A, B)
    
    # The custom kernel computes a different result due to:
    #  - missing the term sum(q*log(q))
    #  - dividing by total number of elements instead of batch_size.
    # Hence, we expect the result to differ from the expected value.
    assert not torch.allclose(kernel_out, expected, atol=1e-5), \
        "Test should fail since the kernel's KL divergence formula is incorrect."

# Issue 3: The kernel does not support tensor types other than float32.
def test_input_tensor_wrong_dtype():
    # Create double precision tensors.
    batch_size = 4
    numel = 8
    A = torch.rand(batch_size, numel, device="cuda", dtype=torch.float64).softmax(dim=-1)
    B = torch.rand(batch_size, numel, device="cuda", dtype=torch.float64).softmax(dim=-1)
    log_A = torch.log(A)
    
    kl_div_module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # This should raise an error or produce an unexpected behavior,
        # because the kernel only handles float (float32) inputs.
        _ = kl_div_module.forward(log_A, B)

# Issue 4: Redundant and improperly implemented warp-level reduction.
def test_incorrect_reduction():
    # Create inputs where the block reduction would be most stressed.
    # Use a size not a multiple of BLOCK_SIZE to force grid-stride looping.
    batch_size = 32
    numel = 1000  # not a multiple of 256; triggers irregular block count reductions.
    A = torch.rand(batch_size, numel, device="cuda", dtype=torch.float32).softmax(dim=-1)
    B = torch.rand(batch_size, numel, device="cuda", dtype=torch.float32).softmax(dim=-1)
    log_A = torch.log(A)
    
    kl_div_module = build_kernel()
    kernel_out = kl_div_module.forward(log_A, B)
    
    # Here, we compare the custom kernel's output against what we would expect from a
    # correctly reduced sum over all elements divided by batch_size. Since the reduction
    # is implemented incorrectly, the output will be off.
    # We compute a "reference" (although the kernel formula is wrong, the reduction error is independent):
    full_sum = 0.0
    # Given the intended (but incorrect) operation the kernel does:
    # Sum_i [exp(log_A[i]) - B[i]*log_A[i]], and then divides by n.
    # We simulate that to see if the reduction itself is wrong.
    log_A_flat = log_A.view(-1)
    B_flat = B.view(-1)
    ref = (torch.exp(log_A_flat) - B_flat * log_A_flat).sum() / B.numel()
    
    # We require that the reduction does not match the properly reduced value.
    assert not torch.allclose(kernel_out, ref, atol=1e-5), \
        "Test should fail due to the incorrect warp-level reduction implementation."

if __name__ == "__main__":
    pytest.main([__file__])
