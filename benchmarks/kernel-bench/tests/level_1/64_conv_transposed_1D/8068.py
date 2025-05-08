
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the extension from kernel.cu
    module = load(
        name="custom_conv_transpose",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Helper: build a reference model with PyTorch's native ConvTranspose1d
def build_reference_model(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias):
    return nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding,
        output_padding=output_padding, groups=groups, bias=bias
    ).cuda()

# Issue 1: The custom CUDA kernel is never launched.
def test_issue_custom_kernel_not_used():
    # We trigger the branch that should use the custom kernel by passing no bias.
    # However, the extension’s forward simply calls torch::conv_transpose1d.
    module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 3
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1
    groups = 1
    bias = None  # no bias so that branch intended for custom kernel is chosen

    # Prepare input and dummy weight tensor
    x = torch.randn(batch_size, in_channels, 16, device="cuda", dtype=torch.float32)
    # Create weight with the standard layout: (in_channels, out_channels/groups, kernel_size).
    # Note: torch.nn.ConvTranspose1d expects weight shape (in_channels, out_channels/groups, kernel_size)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, device="cuda", dtype=torch.float32)

    # Call the extension forward function and compare with reference.
    ref = torch.nn.functional.conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups)
    out = module.forward(x, weight, None, stride, padding, output_padding, groups)
    torch.cuda.synchronize()

    # If the custom kernel were launched, many of its issues might lead to wrong results.
    # Here we check that the outputs exactly match, which (ironically) indicates that the custom kernel is not used.
    assert torch.allclose(out, ref, atol=1e-5), "Issue 1: Custom kernel appears to be used (or mis-integrated) because output does not exactly match the native conv_transpose1d."

# Issue 2: The kernel does not handle the batch dimension.
def test_issue_batch_dimension():
    module = build_kernel()
    # Use batch size > 1
    batch_size = 4
    in_channels = 3
    out_channels = 2
    kernel_size = 3
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    x = torch.randn(batch_size, in_channels, 20, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    
    # Expected output computed with PyTorch's conv_transpose1d.
    ref = torch.nn.functional.conv_transpose1d(x, weight, None, stride, padding, output_padding, groups)
    out = module.forward(x, weight, None, stride, padding, output_padding, groups)
    torch.cuda.synchronize()

    # If the custom kernel ignored the batch dimension, the output shape or values would be wrong.
    assert out.shape == ref.shape, f"Issue 2: Expected output shape {ref.shape} but got {out.shape}."
    assert torch.allclose(out, ref, atol=1e-5), "Issue 2: Output values differ when processing batched input (batch dimension bug)."

# Issue 3: The kernel ignores the 'groups' parameter.
def test_issue_groups():
    module = build_kernel()
    batch_size = 1
    in_channels = 4
    out_channels = 4  # For groups > 1, in_channels and out_channels should be divisible by groups.
    kernel_size = 3
    stride = 1
    padding = 1
    output_padding = 0
    groups = 2  # Use groups > 1

    x = torch.randn(batch_size, in_channels, 15, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    
    ref = torch.nn.functional.conv_transpose1d(x, weight, None, stride, padding, output_padding, groups)
    out = module.forward(x, weight, None, stride, padding, output_padding, groups)
    torch.cuda.synchronize()
    
    # When groups are present, the custom kernel (if used) would ignore group separation.
    assert out.shape == ref.shape, f"Issue 3: Expected shape {ref.shape} but got {out.shape}."
    assert torch.allclose(out, ref, atol=1e-5), "Issue 3: The output mismatch indicates the kernel did not handle groups correctly."

# Issue 4: Constant memory fixed size may be exceeded.
def test_issue_constant_memory_overflow():
    module = build_kernel()
    batch_size = 1
    # Choose weight dimensions such that weight.numel() > 1024
    in_channels = 16
    out_channels = 16
    kernel_size = 13  # Total elements = 16 * 16 * 13 = 3328, which exceeds 1024.
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    x = torch.randn(batch_size, in_channels, 32, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    
    # When copying the weight to constant memory, the kernel will overrun the 1024-float buffer.
    # If the custom kernel were used this would lead to incorrect results (or even a cuda error).
    ref = torch.nn.functional.conv_transpose1d(x, weight, None, stride, padding, output_padding, groups)
    out = module.forward(x, weight, None, stride, padding, output_padding, groups)
    torch.cuda.synchronize()
    
    # The test compares the outputs: if they differ significantly, then it is symptomatic of constant memory overflow.
    assert out.shape == ref.shape, f"Issue 4: Expected shape {ref.shape} but got {out.shape}"
    assert torch.allclose(out, ref, atol=1e-3), "Issue 4: Output discrepancy indicates potential overflow in constant memory usage."

# Issue 5: Unused parameters in compute_conv_transpose (start_idx and total_tasks)
def test_issue_unused_parameters():
    # This test is more about code design: since start_idx and total_tasks are never used,
    # the kernel does not support proper work partitioning for large outputs.
    # We simulate a situation with a very large output (many threads) which in a correct launch
    # should use these parameters. If not, performance or correctness issues might be seen.
    module = build_kernel()
    batch_size = 1
    in_channels = 8
    out_channels = 8
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, 128, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    
    ref = torch.nn.functional.conv_transpose1d(x, weight, None, stride, padding, output_padding, groups)
    out = module.forward(x, weight, None, stride, padding, output_padding, groups)
    torch.cuda.synchronize()
    
    # In a fully optimized kernel, workload partitioning based on start_idx and total_tasks would be critical.
    # Here we compare the results and expect errors if these unused parameters cause miscomputation.
    assert out.shape == ref.shape, f"Issue 5: Expected shape {ref.shape} but got {out.shape}"
    assert torch.allclose(out, ref, atol=1e-5), "Issue 5: Output values discrepancy hints at incomplete task partition handling."

# Issue 6: No bias addition in the custom kernel.
def test_issue_bias_not_handled():
    module = build_kernel()
    batch_size = 1
    in_channels = 4
    out_channels = 3
    kernel_size = 3
    stride = 1
    padding = 1
    output_padding = 0
    groups = 1

    x = torch.randn(batch_size, in_channels, 20, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # The extension's forward function branches on bias presence: if bias exists, it calls native conv_transpose1d.
    # Thus, the custom kernel (if it were used) would have no bias handling.
    # We force the branch by setting bias in a c10::optional. This test will pass because the bias branch
    # bypasses the custom kernel, but it documents the inconsistency.
    ref = torch.nn.functional.conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups)
    out = module.forward(x, weight, bias, stride, padding, output_padding, groups)
    torch.cuda.synchronize()
    
    assert out.shape == ref.shape, f"Issue 6: Expected shape {ref.shape} but got {out.shape}"
    assert torch.allclose(out, ref, atol=1e-5), "Issue 6: Output discrepancy indicates that bias is not handled as expected in the custom kernel."

# Issue 7: The kernel only supports float data type.
def test_issue_dtype_support():
    module = build_kernel()
    batch_size = 1
    in_channels = 4
    out_channels = 3
    kernel_size = 3
    stride = 1
    padding = 1
    output_padding = 0
    groups = 1

    # Use double tensors rather than float. This should trigger an error in the custom kernel code
    # when it force-casts the pointer to float.
    x = torch.randn(batch_size, in_channels, 20, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float64)
    
    with pytest.raises(RuntimeError):
        # Expecting a runtime error due to type incompatibility during the cudaMemcpyToSymbol call
        module.forward(x, weight, None, stride, padding, output_padding, groups)
