
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu.
    cuda_module = load(
        name="kl_div_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def cpu_kl_div(log_predictions, targets, reduction='batchmean'):
    # Compute the KL divergence as implemented in PyTorch:
    # loss = sum(targets * (torch.log(targets) - log_predictions))
    # with reduction 'batchmean' meaning division by batch_size.
    # Note: the PyTorch version does not include the constant term associated with
    # target * log(targets) when targets are probabilities, but we mimic the behavior.
    loss = (targets * (torch.log(targets) - log_predictions)).sum(dim=1)
    if reduction == 'batchmean':
        return loss.mean()
    return loss

def test_wrong_kl_div_computation():
    """
    This test provides simple input values in float32 and compares the output
    of the CUDA kernel with a CPU reference computation for KL divergence.
    Because the kernel uses an incorrect formula (exp(lp) - t*lp), the results
    should differ from the correct computation.
    """
    kernel = build_kernel()
    batch_size = 8
    dim = 64

    # Create normalized probability distributions.
    pred = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    pred = pred.softmax(dim=-1)
    targ = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    targ = targ.softmax(dim=-1)

    # Compute log probabilities.
    log_pred = pred.log()

    # Get kernel result.
    kernel_result = kernel.forward(log_pred, targ).item()

    # Compute the correct KL divergence using CPU (reference).
    ref_result = cpu_kl_div(log_pred.cpu(), targ.cpu(), reduction='batchmean').item()

    # The two results should differ due to the wrong computation in kernel.
    assert abs(kernel_result - ref_result) > 1e-3, (
        f"Kernel output ({kernel_result}) unexpectedly close to reference KL divergence ({ref_result})."
    )

def test_incorrect_reduction():
    """
    This test checks that the reduction factor is not being applied correctly.
    For a given large input, the kernel divides the sum by n (total elements) rather
    than by the batch size. We show that scaling the correct KL divergence by 1/dim
    does not match the kernel output.
    """
    kernel = build_kernel()
    batch_size = 16
    dim = 128

    pred = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    pred = pred.softmax(dim=-1)
    targ = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    targ = targ.softmax(dim=-1)

    log_pred = pred.log()

    kernel_result = kernel.forward(log_pred, targ).item()

    # Correct kl_div with 'batchmean' divides by batch_size, not batch_size*dim.
    ref = cpu_kl_div(log_pred.cpu(), targ.cpu(), reduction='batchmean').item()

    # The kernel divides by total number of elements, so its result should be roughly ref/dim.
    expected_kernel = ref / dim

    assert abs(kernel_result - expected_kernel) > 1e-3, (
        f"Kernel reduction appears to be correct ({kernel_result}) when it should be off "
        f"(expected around {expected_kernel} due to division by total elements)."
    )

def test_dtype_support():
    """
    This test passes double precision (float64) tensors to the kernel.
    Because the kernel only supports float32, it should raise a RuntimeError.
    """
    kernel = build_kernel()
    batch_size = 8
    dim = 32

    # Create double precision tensors.
    pred = torch.rand(batch_size, dim, device='cuda', dtype=torch.float64)
    pred = pred.softmax(dim=-1)
    targ = torch.rand(batch_size, dim, device='cuda', dtype=torch.float64)
    targ = targ.softmax(dim=-1)
    log_pred = pred.log()

    with pytest.raises(RuntimeError):
        # This call should fail because the kernel expects float32 pointers.
        _ = kernel.forward(log_pred, targ)

def test_min_macro_usage():
    """
    This test attempts to trigger any issues that might arise from the unqualified 'min' usage.
    By forcing a very large input, we force the block calculation to exceed 1024 if not capped.
    If there is a compile or runtime issue due to the unqualified min, this test will error out.
    Otherwise, it will simply run and produce a result.
    """
    kernel = build_kernel()
    batch_size = 4
    dim = 1 << 16  # very large number to force many blocks if not capped

    pred = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    pred = pred.softmax(dim=-1)
    targ = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    targ = targ.softmax(dim=-1)
    log_pred = pred.log()

    # We don't have an exact expected numerical result here; we just verify that the call runs.
    result = kernel.forward(log_pred, targ)
    torch.cuda.synchronize()  # ensure kernel execution is complete
    assert result.numel() == 1, "Kernel should return a single scalar result."
