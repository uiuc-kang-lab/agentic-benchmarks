
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Ensure the current path is correct.
    src_path = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="balanced_kl",
        sources=[src_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_incorrect_kl_divergence():
    # This test compares the custom kernel output with PyTorch's kl_div.
    # They are expected to differ because the kernel computes KL divergence incorrectly.
    batch_size = 128
    input_shape = (4096,)
    predictions = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float32).softmax(dim=-1)

    mod = build_kernel()    
    log_preds = torch.log(predictions)
    result_kernel = mod.forward(log_preds, targets)
    
    # PyTorch's reference computation.
    result_ref = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction="batchmean")
    
    # The outputs should not match due to the erroneous computation in the kernel.
    assert not torch.allclose(result_kernel, result_ref, atol=1e-5), (
        f"Kernel output unexpectedly matches PyTorch's result: {result_kernel.item()} vs {result_ref.item()}"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_tensor_alignment_dtype():
    # The kernel expects float32 input and assumes alignment for vectorized (float4) loads.
    # Providing double precision inputs or misaligned sizes should trigger an error.
    batch_size = 64
    input_shape = (257,)  # Choose a shape that likely leads to non-16-byte alignment.
    predictions = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float64).softmax(dim=-1)
    targets = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float64).softmax(dim=-1)
    
    mod = build_kernel()
    # Casting to log space is needed but the type is still double.
    log_preds = torch.log(predictions)
    
    with pytest.raises(RuntimeError):
        # This should raise an exception either due to type mismatch or misaligned vectorized load.
        mod.forward(log_preds, targets)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vectorized_remainder_handling():
    # This test creates an input whose number of elements is not an exact multiple of VEC_SIZE (4),
    # forcing the kernel to process the remainder using the scalar path.
    batch_size = 1
    input_shape = (10,)  # 10 is not divisible by 4 (VEC_SIZE), so 2 elements remain.
    predictions = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float32).softmax(dim=-1)
    
    mod = build_kernel()
    log_preds = torch.log(predictions)
    result = mod.forward(log_preds, targets)
    
    # Since the kernel's computation is highly likely to be wrong due to remainder handling,
    # we assert that the result is nonzero, hinting that the kernel attempted to process all elements.
    assert result.item() != 0.0, f"Kernel output is zero, suggesting that remainder processing might be faulty."
