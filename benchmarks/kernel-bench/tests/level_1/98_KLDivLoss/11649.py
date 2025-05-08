
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="adaptive_kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Formula and reduction scaling mismatch.
# The kernel uses a computation formula that does not match torch.nn.functional.kl_div (with reduction='batchmean').
def test_incorrect_formula_and_reduction():
    # Use a batch of inputs so that expected reduction (division by batch size) does not match division by total elements.
    batch_size = 8
    features = 16
    # Create normalized predictions and targets.
    preds = torch.randn(batch_size, features, device="cuda")
    predictions = torch.softmax(preds, dim=-1)
    log_predictions = torch.log(predictions)
    targets = torch.softmax(torch.randn(batch_size, features, device="cuda"), dim=-1)
    
    kernel_module = build_kernel()
    # Kernel output (incorrectly computed) vs. PyTorch native kl_div (which uses formula:
    # KL(target || input) = sum(target * (log(target) - log_predictions)) / batch_size)
    kernel_out = kernel_module.forward(log_predictions, targets)
    expected = torch.nn.functional.kl_div(log_predictions, targets, reduction="batchmean")
    
    # They are not expected to match because both the formula and reduction division factor are wrong.
    assert not torch.allclose(kernel_out, expected, atol=1e-5), \
        f"Kernel result matched expected value unexpectedly! Kernel: {kernel_out.item()}, Expected: {expected.item()}"

# Issue 2: Assumption that the block’s thread count is a multiple of warp size.
# We try to trigger this by providing an input whose total elements (n) is much less than the threads per block.
def test_incorrect_warp_reduction():
    # Using a small vector where n is not a multiple of the warp (or much smaller than the assumed 256 threads)
    n = 50  # intentionally not a multiple of warp size (32) and far less than 256.
    predictions = torch.randn(n, device="cuda")
    predictions = torch.softmax(predictions, dim=0)
    log_predictions = torch.log(predictions)
    targets = torch.randn(n, device="cuda")
    targets = torch.softmax(targets, dim=0)
    
    kernel_module = build_kernel()
    kernel_out = kernel_module.forward(log_predictions, targets)
    
    # Reference computation according to the kernel’s own formula:
    # ref = sum(exp(log_predictions) - targets * log_predictions) / n
    ref = (torch.exp(log_predictions) - targets * log_predictions).sum() / float(n)
    
    # Since the warp reduction in the kernel takes a shortcut that assumes blockDim.x is an exact multiple of warpSize,
    # this small vector may trigger an out-of-bound or partial-warp reduction error. We expect the kernel output to be off.
    assert not torch.allclose(kernel_out, ref, atol=1e-5), \
        f"Warp reduction issue not triggered; Kernel: {kernel_out.item()}, Ref: {ref.item()}"

# Issue 3: Input tensor type is not checked (expects float32).
def test_input_tensor_dtype():
    n = 1024
    # Create inputs with torch.float64.
    predictions = torch.randn(n, device="cuda", dtype=torch.float64)
    predictions = torch.softmax(predictions, dim=0)
    log_predictions = torch.log(predictions)
    targets = torch.randn(n, device="cuda", dtype=torch.float64)
    targets = torch.softmax(targets, dim=0)
    
    kernel_module = build_kernel()
    # If the kernel is not checking for dtype, it may process the data incorrectly (or even crash).
    # We run the kernel and then check that the result is clearly different from a reference computed in float64.
    kernel_out = kernel_module.forward(log_predictions, targets)
    
    # Compute reference with conversion to float64 following the kernel formula:
    ref = (torch.exp(log_predictions) - targets * log_predictions).sum() / float(n)
    
    assert not torch.allclose(kernel_out, ref, atol=1e-5), \
        "Kernel should misbehave (or produce incorrect result) when given float64 inputs."

# Issue 4: Constant memory usage assumes input fits in 1024 floats without robust checking.
def test_constant_memory_boundary():
    # Test with n exactly equal to 1024, which is the limit for the constant memory buffer.
    n = 1024
    predictions = torch.randn(n, device="cuda")
    predictions = torch.softmax(predictions, dim=0)
    log_predictions = torch.log(predictions)
    targets = torch.randn(n, device="cuda")
    targets = torch.softmax(targets, dim=0)
    
    kernel_module = build_kernel()
    kernel_out = kernel_module.forward(log_predictions, targets)
    
    # Compute a reference value according to the kernel’s own formula:
    ref = (torch.exp(log_predictions) - targets * log_predictions).sum() / float(n)
    
    assert not torch.allclose(kernel_out, ref, atol=1e-5), \
        "Constant memory handling issue not detected when n equals 1024."

