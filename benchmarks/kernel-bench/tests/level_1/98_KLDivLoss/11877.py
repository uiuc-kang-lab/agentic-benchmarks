
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper: compute reference KL divergence as defined by torch.nn.functional.kl_div
# Recall that torch.nn.functional.kl_div with reduction='batchmean'
# computes loss = sum(target * (log(target) - log(prediction)))/batch_size
def ref_kl_div(log_predictions, targets):
    # Add a small epsilon to prevent NaNs in log(target)
    eps = 1e-12
    # Compute elementwise: target * (log(target + eps) - log_prediction)
    loss = targets * (torch.log(targets + eps) - log_predictions)
    # PyTorch batchmean divides by the batch size (first dimension)
    batch_size = log_predictions.size(0)
    return loss.sum() / batch_size

# Test to trigger Issue 1 (Incorrect arithmetic: wrong KL divergence formula)
def test_incorrect_formula():
    cuda_module = build_kernel()
    batch_size = 16
    dims = 64
    # create valid log_predictions and targets (softmax ensures proper distributions)
    preds = torch.randn(batch_size, dims, device="cuda")
    preds = preds.softmax(dim=-1)
    log_preds = torch.log(preds)
    targets = torch.randn(batch_size, dims, device="cuda").softmax(dim=-1)
    
    # Call our custom CUDA kernel function
    out_cuda = cuda_module.forward(log_preds, targets)
    # Compute intended reference loss (batchmean)
    loss_ref = ref_kl_div(log_preds, targets)
    
    # They should not match because the kernel has the wrong arithmetic.
    # This assert expects a noticeable mismatch.
    assert not torch.allclose(out_cuda, loss_ref, atol=1e-4), \
        f"Kernel computed loss unexpectedly close to reference. Kernel={out_cuda.item()}, ref={loss_ref.item()}"

# Test to trigger Issue 2 (Normalization mismatch: division by n instead of batch size)
def test_normalization_mismatch():
    cuda_module = build_kernel()
    batch_size = 8
    dims = 128  # so n = 8 * 128 = 1024; but batchmean should divide by 8, not 1024
    preds = torch.randn(batch_size, dims, device="cuda").softmax(dim=-1)
    log_preds = torch.log(preds)
    targets = torch.randn(batch_size, dims, device="cuda").softmax(dim=-1)
    
    out_cuda = cuda_module.forward(log_preds, targets)
    loss_ref = ref_kl_div(log_preds, targets)
    
    # The kernel divides the sum by n whereas reference divides by batch_size.
    # If the kernel were computing the intended sum S, then kernel loss = S/(batch_size*dims)
    # while batchmean loss = S/batch_size, so the kernel loss is lower by a factor of dims.
    ratio = loss_ref / out_cuda
    # Expect ratio to be around dims if the normalization mismatch exists.
    assert torch.abs(ratio - dims) > 1, \
        f"Normalization factor issue not detected. Expected ratio approx {dims}, got {ratio.item()}"

# Test to trigger Issue 3 (Input tensor data type mismatch)
def test_input_dtype_mismatch():
    cuda_module = build_kernel()
    batch_size = 32
    dims = 256
    # Create inputs in double precision, which the kernel does not support.
    preds = torch.randn(batch_size, dims, device="cuda", dtype=torch.float64).softmax(dim=-1)
    log_preds = torch.log(preds)
    targets = torch.randn(batch_size, dims, device="cuda", dtype=torch.float64).softmax(dim=-1)
    
    with pytest.raises(RuntimeError):
        # Expect the kernel to raise an error because it calls data_ptr<float>()
        cuda_module.forward(log_preds, targets)

# Test to trigger Issue 4 (Non-multiple-of-32 block configuration for shared memory reduction)
# This test creates an input whose total number of elements is not a multiple of 32.
# Although the kernel launch uses a fixed 256 threads per block, the reduction in shared memory
# assumes a complete set of warps. If n is not large or structured enough, the extra threads (which compute zero)
# might disturb the warp-level reduction. We check for numerical inconsistency.
def test_nonmultiple_of_32_input():
    cuda_module = build_kernel()
    batch_size = 7  # intentionally not a multiple of 32
    dims = 100   # so total elements = 700, which is not a multiple of 32
    preds = torch.randn(batch_size, dims, device="cuda").softmax(dim=-1)
    log_preds = torch.log(preds)
    targets = torch.randn(batch_size, dims, device="cuda").softmax(dim=-1)
    
    out_cuda = cuda_module.forward(log_preds, targets)
    loss_ref = ref_kl_div(log_preds, targets)
    # As with our earlier tests, due to arithmetic errors and reduction issues, the result should not match.
    assert not torch.allclose(out_cuda, loss_ref, atol=1e-4), \
        f"Kernel output unexpectedly matches reference in non-multiple-of-32 case. Kernel={out_cuda.item()}, ref={loss_ref.item()}"

if __name__ == "__main__":
    pytest.main([__file__])
