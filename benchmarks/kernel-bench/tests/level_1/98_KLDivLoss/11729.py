
import torch
import pytest
from torch.nn.functional import kl_div as torch_kl_div
from torch.utils.cpp_extension import load

# Build and load the CUDA kernel from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="kl_div_cuda_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Incorrect KL divergence formula.
# For uniform distributions, torch.kl_div returns 0 loss because target==predictions.
# The kernel, however, will compute a nonzero value because exp(log_predictions) returns predictions,
# and the term target*log(target) is missing.
def test_incorrect_formula():
    torch.cuda.manual_seed(0)
    batch_size, dim = 128, 4096
    # Create uniform distributions: predictions and targets are identical uniform vectors.
    predictions = torch.full((batch_size, dim), 1.0 / dim, device="cuda", dtype=torch.float32)
    targets = torch.full((batch_size, dim), 1.0 / dim, device="cuda", dtype=torch.float32)
    # Kernel expects log(predictions) as first argument.
    log_predictions = predictions.log()
    module = build_kernel()
    # Call the CUDA kernel.
    out_kernel = module.forward(log_predictions, targets)
    torch.cuda.synchronize()
    # Expected by PyTorch: KL divergence is 0 when distributions are identical.
    # The kernel will produce a nonzero result due to the missing term.
    assert not torch.allclose(out_kernel, torch.tensor(0.0, device="cuda"), atol=1e-5), \
        f"Kernel should not match the correct KL divergence (expected nonzero error) but got {out_kernel.item()}"

# Test case 2: Incorrect normalization.
# Construct an input where the total number of elements (n) is different from the batch size.
# Compare the kernel output normalization (dividing by n) to torch's "batchmean" reduction.
def test_normalization():
    torch.cuda.manual_seed(0)
    batch_size, dim = 32, 256  # total elements = 32*256 = 8192, but batchmean should divide by 32.
    predictions = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32).softmax(dim=-1)
    log_predictions = predictions.log()
    module = build_kernel()
    out_kernel = module.forward(log_predictions, targets)
    torch.cuda.synchronize()
    # Compute expected loss using PyTorch's kl_div (which uses batchmean: divide by batch size).
    expected = torch_kl_div(log_predictions, targets, reduction="batchmean")
    # Since the kernel divides by the total number of elements, the normalization differs.
    # They should not be close.
    assert not torch.allclose(out_kernel, expected, atol=1e-5), \
        f"Normalization error: kernel output {out_kernel.item()} unexpectedly matches torch.kl_div result {expected.item()}"

# Test case 3: Misaligned input memory.
# Force misaligned memory by slicing off one element so that the underlying data pointer is offset.
def test_misaligned_input():
    torch.cuda.manual_seed(0)
    batch_size, dim = 64, 1025  # 1025 is not a multiple of 4 so that the underlying storage may not be 16-byte aligned
    predictions = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32).softmax(dim=-1)
    # Force misalignment by taking a narrow slice along the last dimension (starting from index 1)
    log_predictions = predictions[:, 1:].log()
    targets_slice = targets[:, 1:]
    module = build_kernel()
    # The kernel does not check alignment, so misaligned inputs may lead to undefined behavior.
    # We simply run the kernel to see if it produces a result (which may be incorrect).
    out_kernel = module.forward(log_predictions, targets_slice)
    torch.cuda.synchronize()
    # There is no “correct” result here; we simply assert that the output is finite.
    assert torch.isfinite(out_kernel).all(), "Kernel produced non-finite output with misaligned inputs"

# Test case 4: Mismatched shapes.
# Provide tensors with different number of elements. The kernel does not check shapes and will use log_predictions.numel().
def test_mismatched_shapes():
    torch.cuda.manual_seed(0)
    batch_size, dim = 32, 128
    predictions = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, dim + 10, device="cuda", dtype=torch.float32).softmax(dim=-1)
    log_predictions = predictions.log()
    module = build_kernel()
    # Calling the kernel with mismatched shapes might lead to a wrong computation or memory over-read.
    # We wrap the call in pytest.raises to catch any potential error.
    with pytest.raises(RuntimeError):
        _ = module.forward(log_predictions, targets)

if __name__ == "__main__":
    pytest.main([__file__])
