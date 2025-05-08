
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the extension module from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Incorrect KL divergence formula.
def test_incorrect_kl_formula():
    # Use a simple case where correct KL divergence should be 0.
    # Consider a uniform distribution:
    # predictions = [0.25, 0.25, 0.25, 0.25]
    # log_predictions = log(predictions)
    # targets = same uniform distribution.
    # torch.nn.functional.kl_div returns 0 when distributions match.
    batch_size = 1
    input_len = 4
    predictions = torch.full((batch_size, input_len), 0.25, device='cuda', dtype=torch.float32)
    log_predictions = torch.log(predictions)
    targets = predictions.clone()
    
    # Expected divergence (batchmean) is 0.
    expected = torch.zeros(1, device='cuda', dtype=torch.float32)
    
    mod = build_kernel()
    # Call the custom kernel
    out = mod.forward(log_predictions, targets)
    torch.cuda.synchronize()
    
    # Due to the wrong formula, the kernel should not return zero.
    assert not torch.allclose(out, expected, atol=1e-5), (
        f"Test failed: Kernel returned {out.item()} which is close to 0, but should be non-zero due to wrong formula."
    )

# Issue 2: Wrong reduction semantics (dividing by total number of elements instead of batch size).
def test_wrong_reduction():
    # Create a batch with batch_size > 1; using the same distribution in each sample
    # For correct batchmean reduction, the total sum divided by batch size should be larger than
    # dividing by total number of elements.
    batch_size = 8
    input_len = 16
    predictions = torch.full((batch_size, input_len), 0.1, device='cuda', dtype=torch.float32)
    # Normalize to make valid probability distribution.
    predictions = predictions / predictions.sum(dim=1, keepdim=True)
    log_predictions = torch.log(predictions)
    targets = predictions.clone()
    
    # Using torch.nn.functional.kl_div with reduction 'batchmean' should yield 0 when distributions match.
    expected = torch.zeros(1, device='cuda', dtype=torch.float32)
    
    mod = build_kernel()
    out = mod.forward(log_predictions, targets)
    torch.cuda.synchronize()
    
    # Even though the distributions match, because of the different scaling we expect an incorrect result (non-zero)
    # from the custom kernel.
    assert not torch.allclose(out, expected, atol=1e-5), (
        f"Test failed: Kernel reduction error not detected. Got {out.item()} which is close to expected {expected.item()}."
    )

# Issue 3: Lack of input tensor type checking (only accepts float32).
def test_input_tensor_type():
    # Create inputs as float64 instead of float32.
    batch_size = 16
    input_len = 32
    predictions = torch.randn(batch_size, input_len, device='cuda', dtype=torch.float64).softmax(dim=-1)
    log_predictions = torch.log(predictions)
    targets = predictions.clone()
    
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel extension to complain or fail when given double tensors,
        # since kernel expects float (float32) pointers.
        mod.forward(log_predictions, targets)

# Issue 4: Assumption of contiguous input tensors.
def test_non_contiguous_input():
    # Create a tensor and then make it non-contiguous by transposing or slicing.
    batch_size = 4
    input_len = 64
    predictions = torch.randn(batch_size, input_len, device='cuda', dtype=torch.float32).softmax(dim=-1)
    log_predictions = torch.log(predictions)
    targets = predictions.clone()
    
    # Make tensors non-contiguous by transposing an extra dimension (e.g., adding an extra dummy dimension)
    log_predictions_noncontig = log_predictions.t()  # Now shape is (input_len, batch_size)
    targets_noncontig = targets.t()
    
    mod = build_kernel()
    
    # We expect the kernel to potentially produce a wrong result or error.
    # If no error is raised, the output may be incorrect. Use a sanity check.
    try:
        out = mod.forward(log_predictions_noncontig.contiguous(), targets_noncontig.contiguous())
        torch.cuda.synchronize()
    except Exception as e:
        pytest.skip("Kernel did not support non-contiguous inputs as expected; exception raised.")
    
    # Now call without forcing contiguous memory.
    with pytest.raises(RuntimeError):
        # This should normally fail or compute incorrect values if the kernel does not properly handle non-contiguous memory.
        mod.forward(log_predictions_noncontig, targets_noncontig)

if __name__ == '__main__':
    pytest.main([__file__])
