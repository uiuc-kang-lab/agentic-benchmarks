
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the CUDA extension module from the file kernel.cu.
    module = load(
        name="masked_cumsum_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

def test_double_type_issue():
    # Issue 1: Using double type (float64) will incorrectly use the Float4 vectorized load.
    batch = 16
    L = 32  # any L divisible by 4 so that the vectorized code path is taken
    x = torch.randn(batch, L, dtype=torch.double, device="cuda")
    mask = torch.randint(0, 2, (batch, L), dtype=torch.bool, device="cuda")
    module = build_kernel()
    result = module.forward(x, mask)
    # Compute reference via PyTorch for the masked cumsum.
    ref = torch.cumsum(x * mask, dim=1)
    # Expect the outputs to differ because of the double type mis-handling.
    assert not torch.allclose(result, ref, atol=1e-4), (
        "Kernel incorrectly handles double types: result matches reference when it should not."
    )

def test_bool_vectorized_issue():
    # Issue 2: The reinterpret_cast for booleans is unsafe.
    # Create an input such that the entire row is processed via the vectorized branch.
    batch = 10
    L = 8  # divisible by 4
    x = torch.arange(1, L + 1, dtype=torch.float32, device="cuda").unsqueeze(0).repeat(batch, 1)
    # Create a nontrivial boolean mask (e.g. alternating True and False)
    mask_pattern = torch.tensor([True, False] * (L // 2), dtype=torch.bool, device="cuda")
    mask = mask_pattern.unsqueeze(0).repeat(batch, 1)
    module = build_kernel()
    result = module.forward(x, mask)
    ref = torch.cumsum(x * mask, dim=1)
    # Because of potential undefined behavior in vectorized bool loads, we expect a mismatch.
    assert not torch.allclose(result, ref, atol=1e-4), (
        "Kernel bool vectorized load issue not triggered: result matches expected output."
    )

def test_no_kernel_launch_error_check():
    # Issue 3: The kernel does not check for launch errors.
    # Here we pass non-contiguous tensors (which should be caught by CHECK_INPUT)
    # so that an error is raised. This indirectly shows that errors in kernel launch
    # would otherwise be silent.
    x = torch.randn(64, 128, dtype=torch.float32, device="cuda").t()  # transpose produces non-contiguous tensor
    mask = torch.randint(0, 2, x.shape, dtype=torch.bool, device="cuda")
    module = build_kernel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        module.forward(x, mask)
