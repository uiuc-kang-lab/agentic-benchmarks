
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 & 2: Test non-divisible size leading to out-of-bound accesses or unhandled tail elements.
def test_non_divisible_size():
    my_module = build_kernel()
    batch_size = 3
    # Create a tensor whose last dimension is not a multiple of 4.
    n = 10  # 10 is not divisible by 4.
    log_predictions = torch.log_softmax(torch.randn(batch_size, n, device="cuda"), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, n, device="cuda"), dim=-1)
    # When the kernel overruns the allocated memory, it should throw a CUDA error.
    with pytest.raises(RuntimeError):
        out = my_module.forward(log_predictions, targets)
        torch.cuda.synchronize()

# Issue 3: Incorrect KL divergence formula.
def test_incorrect_kl_formula():
    my_module = build_kernel()
    batch_size = 4
    n = 8  # Use a size divisible by 4 to avoid triggering other issues.
    log_predictions = torch.log_softmax(torch.randn(batch_size, n, device="cuda"), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, n, device="cuda"), dim=-1)
    out_cuda = my_module.forward(log_predictions, targets)
    # Compute expected result using PyTorch's built-in function.
    expected = torch.nn.functional.kl_div(log_predictions, targets, reduction="batchmean")
    # Because the kernel formula is wrong, the result should not match.
    assert not torch.allclose(out_cuda, expected, atol=1e-4), \
        "Kernel unexpectedly produced the correct KL divergence despite an incorrect implementation."

# Issue 4: Reduction normalization error.
def test_reduction_normalization():
    my_module = build_kernel()
    batch_size = 2
    n = 8  # Divisible by 4.
    log_predictions = torch.log_softmax(torch.randn(batch_size, n, device="cuda"), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, n, device="cuda"), dim=-1)
    out_cuda = my_module.forward(log_predictions, targets)
    # PyTorch's batchmean divides by the batch size (or sums and then divides by batch dim),
    # but the kernel divides by the total element count.
    expected = torch.nn.functional.kl_div(log_predictions, targets, reduction="batchmean")
    assert not torch.allclose(out_cuda, expected, atol=1e-4), \
        "Kernel normalization division is unexpectedly correct; it should differ due to hard-coded division by n."

# Issue 5: Misaligned tensor memory.
def test_misaligned_tensor():
    my_module = build_kernel()
    batch_size = 4
    n = 16  # Use a number divisible by 4.
    # Create a larger tensor and then slice it so that the underlying data pointer is not 16-byte aligned.
    base_log_preds = torch.log_softmax(torch.randn(batch_size, n + 1, device="cuda"), dim=-1)
    base_targets = torch.softmax(torch.randn(batch_size, n + 1, device="cuda"), dim=-1)
    log_predictions = base_log_preds[:, 1:]
    targets = base_targets[:, 1:]
    # Misaligned accesses are likely to cause a CUDA error.
    with pytest.raises(RuntimeError):
        out = my_module.forward(log_predictions, targets)
        torch.cuda.synchronize()

# Issue 6: Hard-coded launch configuration.
def test_hardcoded_launch_configuration():
    my_module = build_kernel()
    batch_size = 128
    # Choose an input size that is not “friendly” to the fixed launch configuration.
    n = 4097  # Not divisible by 4 and not matching the assumed grid dimensions.
    log_predictions = torch.log_softmax(torch.randn(batch_size, n, device="cuda"), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, n, device="cuda"), dim=-1)
    # The kernel may fail or produce incorrect results because of the fixed configuration.
    with pytest.raises(RuntimeError):
        out = my_module.forward(log_predictions, targets)
        torch.cuda.synchronize()
