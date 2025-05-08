
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

def test_incorrect_formula():
    # Issue 1: The kernel computes an incorrect KL divergence.
    # When predictions and targets are identical uniform distributions,
    # the correct kl_div (with reduction="batchmean") should yield 0.
    kernel_module = build_kernel()
    batch_size = 32
    num_classes = 10
    # Uniform distributions
    predictions = torch.full((batch_size, num_classes), 1.0/num_classes, device="cuda", dtype=torch.float32)
    targets = torch.full((batch_size, num_classes), 1.0/num_classes, device="cuda", dtype=torch.float32)
    # Using torch.log(predictions) to mimic the expected input to kl_div
    output = kernel_module.forward(torch.log(predictions), targets)
    torch.cuda.synchronize()
    # The expected correct value is 0, but the kernel's incorrect formula will not yield 0.
    assert not torch.isclose(output, torch.tensor(0.0, device="cuda"), atol=1e-5), \
        f"Kernel incorrectly computed zero loss despite wrong formula."

def test_incorrect_normalization():
    # Issue 2: The kernel divides the sum by total number of elements (n)
    # instead of dividing by the batch size as expected by reduction="batchmean".
    kernel_module = build_kernel()
    batch_size = 16
    num_classes = 8
    predictions = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    output_kernel = kernel_module.forward(torch.log(predictions), targets)
    # Compute correct loss with PyTorch's built-in function (which divides by batch_size)
    output_pytorch = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    torch.cuda.synchronize()
    # The outputs should differ because of the wrong normalization.
    assert not torch.allclose(output_kernel, output_pytorch, atol=1e-5), \
        f"Kernel loss matches PyTorch result unexpectedly despite expected normalization issue."

def test_invalid_input_dtype():
    # Issue 3: The kernel assumes float32 input.
    kernel_module = build_kernel()
    batch_size = 32
    num_classes = 10
    # Create double tensors which are not supported by the kernel.
    predictions = torch.softmax(torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float64), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float64), dim=-1)
    with pytest.raises(RuntimeError):
        kernel_module.forward(torch.log(predictions), targets)

def test_non_contiguous_input():
    # Issue 3: The kernel assumes contiguous input.
    kernel_module = build_kernel()
    batch_size = 16
    num_classes = 8
    predictions = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    # Make the tensors non-contiguous by transposing them.
    predictions = predictions.t()
    targets = targets.t()
    with pytest.raises(RuntimeError):
        kernel_module.forward(torch.log(predictions), targets)

def test_kernel_launch_error_with_large_input():
    # Issue 4: No kernel launch error checking.
    # Create a huge input to increase the likelihood of a launch failure or misconfiguration.
    kernel_module = build_kernel()
    # The size is chosen arbitrarily to stress the kernel launch.
    batch_size = 1
    num_classes = 1 << 24  # very large number of elements
    predictions = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    # Depending on the GPU and available resources, this may trigger a launch error.
    with pytest.raises(Exception):
        _ = kernel_module.forward(torch.log(predictions), targets)

def test_block_size_heuristic():
    # Issue 5: The dynamic block size selection might not be general.
    # Use a tensor whose total element count is not a multiple of the block sizes used (128, 256, or 512).
    kernel_module = build_kernel()
    batch_size = 7
    num_classes = 13  # 7 * 13 = 91, which forces best_block_size = 128 due to n < 8192.
    predictions = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    output_kernel = kernel_module.forward(torch.log(predictions), targets)
    # Compare with PyTorch's implementation (which uses batchmean normalization),
    # so even aside from the normalization and formula issues, the results will differ.
    output_pytorch = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    torch.cuda.synchronize()
    assert not torch.allclose(output_kernel, output_pytorch, atol=1e-5), \
        "Kernel output unexpectedly matches PyTorch's result despite block selection heuristic issues."
