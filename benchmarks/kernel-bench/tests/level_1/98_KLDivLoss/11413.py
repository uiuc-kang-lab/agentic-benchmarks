
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="kl_div_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def module():
    return build_kernel()

# Issue 1: Wrong mathematical formula.
# This test creates simple known tensors where the correct kl_div (as computed by torch.nn.functional.kl_div)
# is different from what our CUDA kernel computes.
def test_incorrect_formula(module):
    # Create a small tensor so that we can compute the reference manually.
    # We choose values that avoid numerical issues.
    preds = torch.tensor([[0.2, 0.3, 0.5]], device="cuda", dtype=torch.float32)
    targets = torch.tensor([[0.1, 0.4, 0.5]], device="cuda", dtype=torch.float32)
    log_preds = torch.log(preds)

    # Compute PyTorch reference using F.kl_div (which expects log_probs and p-probs)
    ref = torch.nn.functional.kl_div(log_preds, targets, reduction='batchmean')
    # Run the custom CUDA kernel
    out = module.forward(log_preds, targets)
    # The kernel output is wrong per issue 1 so we expect a substantial difference.
    # The test passes if the outputs differ significantly.
    assert not torch.allclose(ref, out, atol=1e-4), (
        "Test failure: The kernel output matches the reference unexpectedly. "
        "The kernel appears to use the correct formula, but it should not."
    )

# Issue 2: Incorrect reduction factor (division by n rather than batch size).
def test_incorrect_reduction(module):
    # Create a batch with two samples having 4 elements each.
    # For batchmean, the expected result should be sum(losses)/batch_size (i.e. 2), but our kernel divides by total n (i.e. 8).
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                          [0.3, 0.3, 0.2, 0.2]], device="cuda", dtype=torch.float32)
    targets = torch.tensor([[0.4, 0.3, 0.2, 0.1],
                            [0.1, 0.2, 0.3, 0.4]], device="cuda", dtype=torch.float32)
    log_preds = torch.log(preds)

    ref = torch.nn.functional.kl_div(log_preds, targets, reduction='batchmean')
    out = module.forward(log_preds, targets)
    # Because the kernel divides by n (=8) instead of batch size (=2), the output will be off by a factor of 4.
    assert not torch.allclose(ref, out, atol=1e-4), (
        "Test failure: The kernel reduction factor appears correct, but it should divide by batch size and not total n."
    )

# Issue 3: Input alignment dependency.
def test_alignment_issue(module):
    # Create a tensor and then produce a misaligned view by using storage_offset.
    base = torch.randn(129, device="cuda", dtype=torch.float32)
    # Slice the tensor so that the underlying data pointer is offset by one element –
    # this likely breaks the assumption of 16-byte alignment required for float4 loads.
    misaligned = base.narrow(0, 1, 128)
    # Use softmax to simulate probabilities.
    preds = torch.softmax(misaligned, dim=0).view(1, -1)
    targets = torch.softmax(misaligned.flip(0), dim=0).view(1, -1)
    log_preds = torch.log(preds)

    # The kernel is expected to read misaligned memory with vectorized loads.
    # We test that the kernel either fails or produces a result different from the expected PyTorch result.
    ref = torch.nn.functional.kl_div(log_preds, targets, reduction='batchmean')
    out = module.forward(log_preds, targets)
    assert not torch.allclose(ref, out, atol=1e-4), (
        "Test failure: The kernel did not detect a misaligned input. "
        "It may be incorrectly assuming 16-byte alignment."
    )

# Issue 4: Handling tail elements using min() without proper device definition.
def test_non_multiple_of_vector_size(module):
    # The kernel uses vectorized processing with VEC_SIZE=4.
    # This test ensures that when the total number of elements is not exactly divisible by 4,
    # the branch handling the remainder (with a call to min) is executed.
    # We use an input tensor where numel() % 4 is not zero.
    # For instance, a tensor with 130 elements.
    preds = torch.randn(130, device="cuda", dtype=torch.float32)
    preds = torch.softmax(preds, dim=0).view(1, -1)
    targets = torch.randn(130, device="cuda", dtype=torch.float32)
    targets = torch.softmax(targets, dim=0).view(1, -1)
    log_preds = torch.log(preds)

    ref = torch.nn.functional.kl_div(log_preds, targets, reduction='batchmean')
    out = module.forward(log_preds, targets)
    assert not torch.allclose(ref, out, atol=1e-4), (
        "Test failure: The kernel output is as expected for non-multiple-of-vector-size input. "
        "However, it should trigger the tail processing branch which may be implemented incorrectly."
    )
