
import math
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility: Build the CUDA extension module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="kl_div_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger the incorrect KL divergence formula
def test_incorrect_formula():
    # Use a simple example where predictions and targets are known.
    # We use a 1D tensor with 2 elements and use softmax so that probabilities are valid.
    logits = torch.tensor([[0.0, 0.0]], device="cuda", dtype=torch.float32)
    # predictions after softmax -> [0.5, 0.5]
    predictions = torch.softmax(logits, dim=-1)
    # Let targets be identical so that the correct KL divergence should be 0.
    targets = predictions.clone()
    # Compute expected loss using PyTorch’s kl_div (which computes target*(log(target)-log(prediction)))
    # When predictions == targets, the loss is 0.
    expected_loss = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    my_module = build_kernel()
    kernel_loss = my_module.forward(torch.log(predictions), targets)
    # Because our kernel computes: predictions - target*log(predictions) element-wise,
    # in our case each element becomes: 0.5 - 0.5*log(0.5) = 0.5 + 0.5*0.6931 = ~0.8466.
    # So the final result (divided by n) will definitely differ from 0.
    assert not math.isclose(kernel_loss.item(), expected_loss.item(), rel_tol=1e-5), \
        f"Kernel loss ({kernel_loss.item()}) unexpectedly matches expected loss ({expected_loss.item()})."

# Test case 2: Trigger the wrong normalization factor (batchmean vs total elements)
def test_wrong_normalization():
    # Create an input where the total number of elements is not equal to the batch size.
    # For instance, use a batch of 4 with 10 features each.
    batch_size = 4
    features = 10
    logits = torch.randn(batch_size, features, device="cuda", dtype=torch.float32)
    predictions = torch.softmax(logits, dim=-1)
    targets = torch.softmax(torch.randn(batch_size, features, device="cuda", dtype=torch.float32), dim=-1)
    # Expected loss is computed by torch.nn.functional.kl_div with reduction='batchmean'
    expected_loss = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    my_module = build_kernel()
    kernel_loss = my_module.forward(torch.log(predictions), targets)
    # Because the kernel divides by total number of elements (batch_size*features)
    # while the expected loss divides by batch_size, the two values should differ.
    assert not math.isclose(kernel_loss.item(), expected_loss.item(), rel_tol=1e-5), \
        f"Kernel loss ({kernel_loss.item()}) incorrectly normalized and matches expected loss ({expected_loss.item()})."

# Test case 3: Trigger the issue with unsupported tensor data type (double)
def test_invalid_dtype():
    # Use double precision tensors which the kernel does not support.
    batch_size = 8
    features = 16
    logits = torch.randn(batch_size, features, device="cuda", dtype=torch.float64)
    predictions = torch.softmax(logits, dim=-1)
    targets = torch.softmax(torch.randn(batch_size, features, device="cuda", dtype=torch.float64), dim=-1)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel expects float32. This should raise an error.
        _ = my_module.forward(torch.log(predictions), targets)
    # If no error is raised, the test will fail.

# Test case 4: Trigger issues with non-contiguous input tensors.
def test_noncontiguous():
    # Create contiguous tensors first
    batch_size = 8
    features = 16
    logits = torch.randn(batch_size, features, device="cuda", dtype=torch.float32)
    predictions = torch.softmax(logits, dim=-1)
    targets = torch.softmax(torch.randn(batch_size, features, device="cuda", dtype=torch.float32), dim=-1)
    # Make them non-contiguous by transposing (if 2D, transpose will make it non-contiguous)
    predictions_noncontig = predictions.t()
    targets_noncontig = targets.t()
    
    my_module = build_kernel()
    # Run kernel using contiguous inputs for baseline
    contiguous_loss = my_module.forward(torch.log(predictions), targets)
    # Run kernel with non-contiguous inputs.
    noncontig_loss = my_module.forward(torch.log(predictions_noncontig), targets_noncontig)
    # The outputs might differ because the kernel does not account for non-contiguous memory.
    # We check that they are not close.
    assert not torch.allclose(contiguous_loss, noncontig_loss, atol=1e-5), \
        "Kernel produced the same output for contiguous and non-contiguous inputs, but may be improperly handling memory layout."

# Test case 5: Trigger the potential issue with block count due to using undefined min macro.
def test_block_count_issue():
    # Construct a large input so that (n + threads - 1)/threads > 1024.
    # This will force the kernel to limit the number of blocks to 1024.
    # Even though the issue is a compile-time one (missing proper include or namespace for min),
    # the wrong block count combined with normalization by total elements can be detected at runtime.
    batch_size = 1000  # large batch size
    features = 4096    # large number of features so that total elements is huge
    logits = torch.randn(batch_size, features, device="cuda", dtype=torch.float32)
    predictions = torch.softmax(logits, dim=-1)
    targets = torch.softmax(torch.randn(batch_size, features, device="cuda", dtype=torch.float32), dim=-1)
    
    my_module = build_kernel()
    kernel_loss = my_module.forward(torch.log(predictions), targets)
    expected_loss = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    # Because of two issues (wrong normalization and forced block count capped at 1024),
    # the kernel output should differ significantly from the expected loss.
    diff = abs(kernel_loss.item() - expected_loss.item())
    assert diff > 1e-3, f"Kernel loss ({kernel_loss.item()}) unexpectedly close to expected loss ({expected_loss.item()}); possible block count issue."

