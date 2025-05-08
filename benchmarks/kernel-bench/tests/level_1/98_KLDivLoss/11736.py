
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def module():
    return build_kernel()

# Test 1: Verify that the computed KL divergence is not matching the expected formula.
def test_incorrect_kl_formula(module):
    # Create small, controlled input where we can compute the expected value.
    # We use a batch size >1 to emulate batchmean reduction.
    batch_size = 4
    n_features = 8
    # Create "predictions" as softmax activations and targets similarly.
    predictions = torch.rand(batch_size, n_features, device="cuda", dtype=torch.float32)
    predictions = torch.softmax(predictions, dim=1)
    targets = torch.rand(batch_size, n_features, device="cuda", dtype=torch.float32)
    targets = torch.softmax(targets, dim=1)
    # Use PyTorch's kl_div (which uses target*(log(target) - log(prediction)))
    log_preds = torch.log(predictions)
    expected = torch.nn.functional.kl_div(log_preds, targets, reduction='batchmean')

    # Use our CUDA kernel (which divides by n and uses a different formula)
    result = module.forward(log_preds, targets)
    torch.cuda.synchronize()
    # The difference in formula should yield a significant difference.
    assert not torch.allclose(result, expected, atol=1e-4), \
        "Test failed: The kernel's computed KL divergence unexpectedly matches the correct value."

# Test 2: The normalization factor is incorrect (dividing by n rather than batch size).
def test_normalization_divisor(module):
    batch_size = 8
    n_features = 16
    predictions = torch.rand(batch_size, n_features, device="cuda", dtype=torch.float32)
    predictions = torch.softmax(predictions, dim=1)
    targets = torch.rand(batch_size, n_features, device="cuda", dtype=torch.float32)
    targets = torch.softmax(targets, dim=1)
    log_preds = torch.log(predictions)
    # PyTorch's kl_div with "batchmean" divides the sum by batch_size.
    expected = torch.nn.functional.kl_div(log_preds, targets, reduction='batchmean')
    
    result = module.forward(log_preds, targets)
    torch.cuda.synchronize()
    # Because our kernel divides by n (batch_size*n_features),
    # the result will be scaled down by a factor of n_features.
    scale_factor = n_features
    assert not torch.allclose(result * scale_factor, expected, atol=1e-4), \
        "Test failed: Kernel normalization unexpectedly matches the batchmean criteria."

# Test 3: Using double precision inputs should trigger an issue.
def test_input_tensor_type(module):
    batch_size = 4
    n_features = 8
    # Create double precision tensors.
    predictions = torch.rand(batch_size, n_features, device="cuda", dtype=torch.float64)
    predictions = torch.softmax(predictions, dim=1)
    targets = torch.rand(batch_size, n_features, device="cuda", dtype=torch.float64)
    targets = torch.softmax(targets, dim=1)
    log_preds = torch.log(predictions)
    with pytest.raises(RuntimeError):
        # Since the kernel only supports float32, this should raise an error.
        _ = module.forward(log_preds, targets)
    torch.cuda.synchronize()

# Test 4: Using inputs with a size not divisible by 4, and/or non-contiguous tensors.
def test_non_divisible_and_noncontiguous(module):
    batch_size = 5    # use a batch size producing a total element count not divisible by 4
    n_features = 7    # 5*7=35, not divisible by 4
    predictions = torch.rand(batch_size, n_features, device="cuda", dtype=torch.float32)
    # Make tensor non-contiguous by transposing after unsqueeze.
    predictions = predictions.unsqueeze(0).transpose(0, 1).squeeze(1)
    predictions = torch.softmax(predictions, dim=-1)
    
    targets = torch.rand(batch_size, n_features, device="cuda", dtype=torch.float32)
    targets = targets.unsqueeze(0).transpose(0, 1).squeeze(1)
    targets = torch.softmax(targets, dim=-1)
    log_preds = torch.log(predictions)
    # Even though the kernel does not check for non-contiguity, this test
    # is expected to trigger misaligned accesses or unexpected behavior.
    result = module.forward(log_preds, targets)
    torch.cuda.synchronize()
    # Rather than comparing to a correct result, we just check that the output is numerically off
    correct = torch.nn.functional.kl_div(torch.log(torch.softmax(predictions, dim=-1)), targets, reduction='batchmean')
    assert not torch.allclose(result, correct, atol=1e-4), \
        "Test failed: The kernel output unexpectedly matches the correct result for non-contiguous inputs."
