
import torch
import pytest
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Note: The .cu file is assumed to be in the current directory.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    module = load(
        name="kl_div_cuda",
        sources=[os.path.join(this_dir, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1 and 2: Wrong KL-divergence formula and incorrect scaling.
def test_incorrect_formula_and_reduction():
    # Create simple distributions that are easy to compute.
    # Use a batch size of 2 and 4 classes, so that we can compute the expected reference.
    # Note: F.kl_div expects input as log_predictions.
    preds = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                          [0.3, 0.3, 0.2, 0.2]], device="cuda", dtype=torch.float32)
    targets = torch.tensor([[0.2, 0.3, 0.1, 0.4],
                            [0.1, 0.4, 0.3, 0.2]], device="cuda", dtype=torch.float32)
    log_preds = torch.log(preds)
    # Compute reference output using PyTorch.
    # Note that F.kl_div with reduction 'batchmean' divides the sum by the batch size.
    ref = F.kl_div(log_preds, targets, reduction='batchmean')
    
    kernel_module = build_kernel()
    out = kernel_module.forward(log_preds, targets)
    # The kernel divides by total number of elements, so expected value would be ref * (batch_size/total_elements)
    adjusted_ref = ref * (log_preds.shape[0] / log_preds.numel())
    
    # Because of the formula difference, the computed kernel output should not match the reference.
    # We require a sufficiently large difference to trigger that the issue is present.
    assert not torch.allclose(out, ref, atol=1e-3), \
        f"Kernel output unexpectedly matches PyTorch F.kl_div. out={out.item()}, ref={ref.item()}"
    # And also check that the scaling is wrong (adjusted_ref is not equal to kernel's output).
    assert not torch.allclose(out, adjusted_ref, atol=1e-3), \
        f"Kernel output matches the adjusted reference. out={out.item()}, adjusted_ref={adjusted_ref.item()}"

# Issue 3: Block reduction assumes threads per block is a multiple of 32.
def test_non_multiple_warp():
    # Create an input where the total number of elements is less than the (hardcoded) block size used in the kernel launch.
    # This forces some threads (among the 256 threads per block) to have no work.
    # In such a scenario, the block-level reduction may be incorrect if the number of active threads is
    # not a multiple of 32.
    N = 100  # Not a multiple of 32
    preds = torch.softmax(torch.randn(N, device="cuda", dtype=torch.float32), dim=0)
    targets = torch.softmax(torch.randn(N, device="cuda", dtype=torch.float32), dim=0)
    log_preds = torch.log(preds).contiguous()  # Ensure contiguous
    kernel_module = build_kernel()
    out = kernel_module.forward(log_preds, targets)
    # Compute a reference using a naive implementation following the expected math.
    # Here we use the PyTorch kl_div to get the intended result.
    ref = F.kl_div(log_preds, targets, reduction='batchmean')
    # They should differ because the reduction in the kernel is done improperly for non-multiple
    # of 32 thread requirements.
    assert not torch.allclose(out, ref, atol=1e-3), \
        f"Kernel reduction did not exhibit the expected error with non-multiple-of-warp block size: out={out.item()}, ref={ref.item()}"

# Issue 4: Input tensor data type is not validated (only float32 is supported).
def test_input_tensor_dtype():
    # Create inputs with double precision.
    preds = torch.softmax(torch.randn(128, 4096, device="cuda", dtype=torch.float64), dim=-1)
    targets = torch.softmax(torch.randn(128, 4096, device="cuda", dtype=torch.float64), dim=-1)
    log_preds = torch.log(preds)
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel expects float32 pointers.
        kernel_module.forward(log_preds, targets)

# Issue 4 (continued): Non-contiguous tensor input.
def test_non_contiguous_input():
    # Create contiguous inputs and then create a non-contiguous view.
    a = torch.softmax(torch.randn(128, 4096, device="cuda", dtype=torch.float32), dim=-1)
    b = torch.softmax(torch.randn(128, 4096, device="cuda", dtype=torch.float32), dim=-1)
    # Create non-contiguous tensors by transposing (note: for 2D tensors, transpose creates non-contiguity)
    a_noncontig = a.t()
    b_noncontig = b.t()
    # Log of a non-contiguous tensor
    log_a_noncontig = torch.log(a_noncontig)
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel likely expects contiguous memory access.
        kernel_module.forward(log_a_noncontig, b_noncontig)
