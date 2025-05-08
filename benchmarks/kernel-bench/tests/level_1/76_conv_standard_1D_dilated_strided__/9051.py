
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

def test_invalid_kernel_range_and_bounds():
    # This test is intended to trigger issues 1 and 2.
    # We choose parameters in which the computation of the valid k range can be fragile.
    # For example, with a high dilation the computed range may allow input_pos to overshoot.
    B = 1
    in_channels = 1
    out_channels = 1
    kernel_size = 5
    length = 10
    stride = 1
    dilation = 3  # relatively high dilation
    # Create input and weight (no bias)
    x = torch.randn(B, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float32)
    # Use built-in conv1d (without padding) as reference.
    ref_output = torch.nn.functional.conv1d(x, weight, bias=None, stride=stride, padding=0, dilation=dilation)
    # Build custom CUDA kernel module.
    module = build_kernel()
    custom_output = module.forward(x, weight, torch.tensor([]).to(x.device), stride, dilation)
    torch.cuda.synchronize()
    # Expect a difference since the computed index ranges may be wrong.
    assert not torch.allclose(custom_output, ref_output, atol=1e-4), (
        "Custom kernel output unexpectedly matches reference output. "
        "Possible miscomputation of valid index range or missing bounds checks."
    )

def test_float_only_support():
    # This test is intended to trigger issue 3.
    # Passing half precision tensors (float16) should either fail or produce incorrect results,
    # since the kernel only supports float32.
    B = 1
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    length = 20
    stride = 1
    dilation = 1
    x = torch.randn(B, in_channels, length, device="cuda", dtype=torch.float16)
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float16)
    # We expect an exception due to type mismatch (or at least wrong results).
    module = build_kernel()
    with pytest.raises(Exception):
        # The kernel’s forward should check (or misbehave) because data_ptr<float>() is used.
        _ = module.forward(x, weight, torch.tensor([]).to(x.device), stride, dilation)

def test_ldg_usage_consistency():
    # This test is intended to trigger issue 4.
    # If __ldg is not appropriate for non-readonly data, repeated runs may produce inconsistent results.
    B = 2
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    length = 32
    stride = 1
    dilation = 1
    x = torch.randn(B, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    module = build_kernel()
    output1 = module.forward(x, weight, bias, stride, dilation)
    output2 = module.forward(x, weight, bias, stride, dilation)
    torch.cuda.synchronize()
    # Here we require that multiple invocations yield identical results.
    assert torch.allclose(output1, output2, atol=1e-6), (
        "Custom kernel produced inconsistent outputs on repeated runs; "
        "this may indicate issues with the __ldg usage."
    )
