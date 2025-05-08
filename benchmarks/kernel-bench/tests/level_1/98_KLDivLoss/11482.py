
import os
import re
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension from the given source filename.
def build_kernel(extra_cuda_cflags=None, source_modification_fn=None):
    # Read the original source file content
    with open("kernel.cu", "r") as f:
        source = f.read()

    # Optionally modify the source code (e.g. for testing block size issues)
    if source_modification_fn is not None:
        source = source_modification_fn(source)

    # Save modified source to a temporary file
    mod_source_filename = "temp_kernel.cu"
    with open(mod_source_filename, "w") as f:
        f.write(source)

    cuda_module = load(
        name="test_module",
        sources=[mod_source_filename],
        extra_cuda_cflags=extra_cuda_cflags if extra_cuda_cflags is not None else ["-O3", "--use_fast_math"],
        verbose=False,
    )

    # Clean up temporary file after load (if desired)
    os.remove(mod_source_filename)
    return cuda_module

# 1. Test for incorrect normalization.
# The kernel divides the accumulated sum by n (total number of elements) rather than the batch size.
def test_normalization():
    # Create a small tensor with known values.
    batch_size = 8
    feature_size = 16
    # Make predictions and targets that sum to 1 along the feature dimension
    predictions = torch.rand(batch_size, feature_size, device='cuda')
    predictions = torch.nn.functional.softmax(predictions, dim=-1)
    targets = torch.rand(batch_size, feature_size, device='cuda')
    targets = torch.nn.functional.softmax(targets, dim=-1)

    # Compute reference using torch.nn.functional.kl_div (batchmean reduction divides by batch_size)
    log_predictions = torch.log(predictions)
    ref = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')

    # Build the module from the kernel file.
    mod = build_kernel()

    # Use the custom CUDA kernel forward.
    out = mod.forward(log_predictions, targets)

    # Because the kernel divides by total number of elements, the value should differ by a factor of feature_size.
    # We expect:
    expected = ref * (feature_size)
    # Use a tolerance and check that they are not equal (i.e. the normalization is wrong).
    assert not torch.allclose(out, ref, atol=1e-5), "Kernel normalization appears correct, but an error was expected."
    # Also check that out is approximately expected * (1/feature_size) off.
    assert torch.allclose(out, expected, atol=1e-3), f"Kernel normalization likely incorrect. Expected {expected.item()} but got {out.item()}."

# 2. Test to trigger the dtype requirement (only float32 accepted)
def test_input_dtype():
    batch_size = 8
    feature_size = 16
    # Create tensors with dtype float64 instead of float32.
    predictions = torch.rand(batch_size, feature_size, device="cuda", dtype=torch.float64)
    predictions = torch.nn.functional.softmax(predictions, dim=-1)
    targets = torch.rand(batch_size, feature_size, device="cuda", dtype=torch.float64)
    targets = torch.nn.functional.softmax(targets, dim=-1)
    log_predictions = torch.log(predictions)

    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel to fail (or produce an error) due to wrong input type.
        mod.forward(log_predictions, targets)
        
# 3. Test to trigger non-contiguous memory issues.
def test_non_contiguous():
    batch_size = 8
    feature_size = 16
    # Create contiguous tensors then make non-contiguous versions by transposing a 2D tensor.
    predictions = torch.rand(batch_size, feature_size, device='cuda').softmax(dim=-1)
    targets = torch.rand(batch_size, feature_size, device='cuda').softmax(dim=-1)
    
    # Make non-contiguous versions: transpose twice so shape remains same but memory layout is changed.
    predictions_noncontig = predictions.t().t()
    targets_noncontig = targets.t().t()
    # Verify non-contiguity
    assert not predictions_noncontig.is_contiguous(), "Expected predictions to be non-contiguous."
    assert not targets_noncontig.is_contiguous(), "Expected targets to be non-contiguous."
    
    log_predictions = torch.log(predictions_noncontig)
    
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel to error out or produce unexpected behavior for non-contiguous tensors.
        mod.forward(log_predictions, targets_noncontig)

# 4. Test to trigger the block size assumption (simulate a blockDim.x smaller than 32).
# Do this by modifying the kernel source to launch with 16 threads instead of 256.
def test_block_size():
    def modify_source(source):
        # Replace the hard-coded thread count "256" with "16" in the host function.
        modified_source = re.sub(r'const int threads = 256;', 'const int threads = 16;', source)
        return modified_source

    batch_size = 8
    feature_size = 16
    predictions = torch.rand(batch_size, feature_size, device='cuda').softmax(dim=-1)
    targets = torch.rand(batch_size, feature_size, device='cuda').softmax(dim=-1)
    log_predictions = torch.log(predictions)
    
    # Build the module with modified launch configuration.
    mod = build_kernel(source_modification_fn=modify_source)
    
    # Run the kernel. In a proper implementation the reduction would work for any block size,
    # but with blockDim.x < 32, our warp-level reduction will access shared memory erroneously.
    # The output is expected to be incorrect.
    out = mod.forward(log_predictions, targets)
    
    # Compute the reference value using torch.kl_div.
    ref = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    # Since issue #2 is still present (normalization error), we compute expected based on the flawed reduction
    # The flawed kernel divides the sum by the total number of elements, so expected value is ref * feature_size.
    expected = ref * feature_size

    # Check that the result does not match the correct batchmean (or expected correct) value.
    assert not torch.allclose(out, ref, atol=1e-5), "Kernel produced output matching batchmean reduction, which is unexpected given block size issue."
    assert torch.allclose(out, expected, atol=1e-3), f"Kernel output {out.item()} does not match the expected flawed reduction value {expected.item()}."

if __name__ == "__main__":
    pytest.main([__file__])
