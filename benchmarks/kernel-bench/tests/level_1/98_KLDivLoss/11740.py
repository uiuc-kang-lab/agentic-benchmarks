
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# 1. Misaligned memory test: Create a tensor slice (non-aligned pointer)
def test_misaligned_input():
    # Allocate a tensor with extra data then slice to force misalignment.
    base = torch.randn(1024 + 1, dtype=torch.float32, device="cuda")
    # Slicing off the first element could lead to an address offset by sizeof(float)
    misaligned = base[1:].contiguous()  # Now misaligned relative to float4 boundaries
    # Create targets similarly misaligned
    base_t = torch.randn(1024 + 1, dtype=torch.float32, device="cuda")
    misaligned_t = base_t[1:].contiguous()
    kl_mod = build_kernel()
    # We expect the kernel may malfunction (e.g., produce a wrong result or crash)
    with pytest.raises(Exception):
        res = kl_mod.forward(misaligned, misaligned_t)
        torch.cuda.synchronize()

# 2. Non-contiguous tensor test: Pass in non-contiguous inputs.
def test_non_contiguous_input():
    # Create a contiguous tensor then perform a transpose to make it non-contiguous.
    a = torch.randn(32, 32, dtype=torch.float32, device="cuda")
    b = torch.randn(32, 32, dtype=torch.float32, device="cuda")
    a_t = a.t()  # Not contiguous
    b_t = b.t()  # Not contiguous
    kl_mod = build_kernel()
    with pytest.raises(Exception):
        res = kl_mod.forward(a_t, b_t)
        torch.cuda.synchronize()

# 3. Incorrect KL divergence formula test: Compare with PyTorch reference computation.
def test_incorrect_formula():
    # Create valid probability distributions (softmaxed)
    N = 512
    pred = torch.softmax(torch.randn((1, N), dtype=torch.float32, device="cuda"), dim=-1)
    targ = torch.softmax(torch.randn((1, N), dtype=torch.float32, device="cuda"), dim=-1)
    # Our forward expects log(predictions) as input, so we mimic that
    log_pred = torch.log(pred)
    kl_mod = build_kernel()
    custom_out = kl_mod.forward(log_pred, targ)
    # PyTorch reference: kl_div expects input as log-probabilities
    ref_out = torch.nn.functional.kl_div(log_pred, targ, reduction='batchmean')
    # They should be very close if the kernel was correct.
    # We expect a noticeable discrepancy because the kernel omits target * log(target)
    assert not torch.allclose(custom_out, ref_out, atol=1e-4), \
        "Kernel KL divergence formula unexpectedly matches the PyTorch kl_div."

# 4. Incorrect scaling factor test:
def test_incorrect_scaling():
    # Use a larger batch
    batch_size = 64
    N = 2048
    preds = torch.softmax(torch.randn(batch_size, N, dtype=torch.float32, device="cuda"), dim=-1)
    targs = torch.softmax(torch.randn(batch_size, N, dtype=torch.float32, device="cuda"), dim=-1)
    log_preds = torch.log(preds)
    kl_mod = build_kernel()
    custom_out = kl_mod.forward(log_preds, targs)
    # PyTorch computes batchmean (sum over batch elements divided by batch size)
    ref_out = torch.nn.functional.kl_div(log_preds, targs, reduction='batchmean')
    # The scaling error (dividing by total elements) should lead to a discrepancy.
    assert not torch.allclose(custom_out, ref_out, atol=1e-4), \
        "Kernel scaling factor appears to be correct when it should differ from torch.nn.functional.kl_div."

# 5. Type checking test: Pass in double tensors.
def test_incorrect_type():
    # Create double precision tensors
    N = 1024
    preds = torch.softmax(torch.randn((1, N), dtype=torch.float64, device="cuda"), dim=-1)
    targs = torch.softmax(torch.randn((1, N), dtype=torch.float64, device="cuda"), dim=-1)
    log_preds = torch.log(preds)
    kl_mod = build_kernel()
    with pytest.raises(Exception):
        res = kl_mod.forward(log_preds, targs)
        torch.cuda.synchronize()

# 6. Excessively large tensor test: Force a number of blocks beyond 1024.
def test_excessive_blocks():
    # Create an input tensor large enough so that the ideal block count exceeds 1024.
    # Each block handles threads * 4 elements. With threads=256, 1025 blocks would be needed for > 1024*256*4 elements.
    num_elems = 1025 * 256 * 4
    # Create a 1D tensor and softmax it (to simulate a probability distribution)
    input_tensor = torch.softmax(torch.randn(num_elems, dtype=torch.float32, device="cuda"), dim=0)
    # Use the same tensor for both log_predictions and targets
    log_preds = torch.log(input_tensor)
    kl_mod = build_kernel()
    # The kernel caps blocks to 1024, so the reduction might not be computed over the full tensor.
    custom_out = kl_mod.forward(log_preds, input_tensor)
    # Compute reference kl_div using torch.nn.functional.kl_div with reduction 'batchmean'
    ref_out = torch.nn.functional.kl_div(log_preds, input_tensor, reduction='batchmean')
    assert not torch.allclose(custom_out, ref_out, atol=1e-4), \
        "Kernel block capping did not affect output scaling as expected."

if __name__ == "__main__":
    pytest.main([__file__])
