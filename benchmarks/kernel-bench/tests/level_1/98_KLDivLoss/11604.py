
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Triggering the wrong computation (formula and reduction factor)
def test_incorrect_formula_and_reduction():
    # Use a tensor with a batch dimension > 1 so that 'batchmean' should divide by batch_size,
    # but the kernel divides by the total number of elements.
    batch_size = 2
    elems = 5
    # Create valid probability distributions.
    predictions = torch.rand(batch_size, elems, device='cuda').softmax(dim=-1)
    targets = torch.rand(batch_size, elems, device='cuda').softmax(dim=-1)
    # The kernel expects log probabilities.
    log_preds = torch.log(predictions)
    
    # Compute the correct KL divergence using PyTorch's built-in function.
    # PyTorch kl_div with reduction='batchmean' computes:
    #   kl = sum(target * (log(target) - log_preds)) / batch_size
    kl_ref = torch.nn.functional.kl_div(log_preds, targets, reduction='batchmean')
    
    kernel_module = build_kernel()
    kl_kernel = kernel_module.forward(log_preds, targets)
    
    # Due to the wrong formula and reduction, the kernel result will be scaled and offset differently.
    diff = (kl_kernel - kl_ref).abs().item()
    msg = (
        f"Kernel output ({kl_kernel.item():.6f}) is too similar to the correct value "
        f"({kl_ref.item():.6f}). Expected a significant discrepancy due to kernel errors. Diff: {diff}"
    )
    assert diff > 1e-3, msg

# Test case 2: Triggering type incompatibility
def test_input_tensor_type():
    batch_size = 128
    elems = 4096
    # Create double precision tensors instead of float32.
    predictions = torch.randn(batch_size, elems, device="cuda", dtype=torch.double).softmax(dim=-1)
    targets = torch.randn(batch_size, elems, device="cuda", dtype=torch.double).softmax(dim=-1)
    
    kernel_module = build_kernel()
    # Expect the kernel to raise a RuntimeError due to incompatible data type 
    # (it expects float32 pointers via data_ptr<float>).
    with pytest.raises(RuntimeError):
        _ = kernel_module.forward(torch.log(predictions), targets)

# Test case 3: Triggering block configuration issue with a thread block not a multiple of 32
def test_non_multiple_of_warp():
    # Though the current host function hardcodes 256 threads (a multiple of 32),
    # we can simulate an input whose size is not a multiple of 32.
    # This stresses the reduction code which assumes full warps.
    batch_size = 1
    elems = 50  # 50 is not a multiple of 32.
    predictions = torch.rand(batch_size, elems, device='cuda').softmax(dim=-1)
    targets = torch.rand(batch_size, elems, device='cuda').softmax(dim=-1)
    log_preds = torch.log(predictions)
    
    # Compute reference KL divergence using PyTorch (note: reduction='batchmean' here divides by batch_size).
    kl_ref = torch.nn.functional.kl_div(log_preds, targets, reduction='batchmean')
    
    kernel_module = build_kernel()
    kl_kernel = kernel_module.forward(log_preds, targets)
    
    # Even if the kernel produces a number without a crash, the result will be numerically off 
    # due to the warp reduction assumptions.
    diff = (kl_kernel - kl_ref).abs().item()
    msg = (
        f"Kernel output ({kl_kernel.item():.6f}) unexpectedly close to the reference value "
        f"({kl_ref.item():.6f}). Expected a significant difference due to block configuration errors. Diff: {diff}"
    )
    assert diff > 1e-3, msg
