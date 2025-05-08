
import pytest
import torch
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

def test_hardsigmoid_clamp_correctness():
    # Issue 1: The kernel’s use of max/min may be ambiguous.
    # This test verifies that the computed HardSigmoid value is correctly clamped.
    # HardSigmoid: output = clamp((x+3)/6, 0, 1)
    input_tensor = torch.tensor([-10.0, -3.0, 0.0, 3.0, 10.0], device="cuda", dtype=torch.float32)
    expected = ((input_tensor + 3.0) / 6.0).clamp(min=0, max=1)
    module = build_kernel()
    output = module.forward(input_tensor)
    torch.cuda.synchronize()
    assert torch.allclose(output, expected, atol=1e-5), \
        f"Output {output} does not match expected value {expected}."

def test_half_precision_not_supported():
    # Issue 2: The kernel does not dispatch for half-precision inputs.
    # This test attempts to run the kernel on a float16 input and expects a RuntimeError.
    input_tensor = torch.randn(1024, device="cuda", dtype=torch.float16)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        module.forward(input_tensor)

def test_small_tensor_launch_bounds():
    # Issue 3: The hard-coded __launch_bounds__(256) may be too restrictive.
    # This test uses a very small tensor (only one element) to see if any occupancy issues arise.
    input_tensor = torch.tensor([0.0], device="cuda", dtype=torch.float32)
    expected = ((input_tensor + 3.0) / 6.0).clamp(min=0, max=1)
    module = build_kernel()
    output = module.forward(input_tensor)
    torch.cuda.synchronize()
    assert torch.allclose(output, expected, atol=1e-5), \
        f"Output {output} does not match expected value {expected}."
