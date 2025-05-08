
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to compile and load the CUDA kernel extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# A helper function to compute reference output using PyTorch's native implementation.
def reference_conv_transpose2d(x, weight, bias, stride, padding):
    conv = torch.nn.ConvTranspose2d(
        in_channels=x.size(1),
        out_channels=weight.size(1),
        kernel_size=(weight.size(2), weight.size(3)),
        stride=stride,
        padding=padding,
        bias=(bias is not None)
    )
    conv.weight.data.copy_(weight)
    if bias is not None:
        conv.bias.data.copy_(bias)
    conv.to(x.device)
    return conv(x)

# Issue 1: #pragma unroll on runtime bounds
# Use non-standard but valid kernel dimensions that are not known at compile time.
def test_unroll_issue():
    cuda_module = build_kernel()
    # Create input with non-standard sizes to force runtime-determined loop bounds.
    N, C_in, H_in, W_in = 2, 7, 15, 16  # C_in, kernel dims are not multiples of typical unroll factors.
    C_out = 3
    kH, kW = 3, 5
    stride = (1, 1)
    padding = (1, 2)

    x = torch.randn(N, C_in, H_in, W_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_in, C_out, kH, kW, device="cuda", dtype=torch.float32)
    bias = None

    out_kernel = cuda_module.forward(x, weight, bias, list(stride), list(padding))
    out_ref = reference_conv_transpose2d(x, weight, bias, stride, padding)
    # They might not match if unroll misbehaves.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-4), \
        "Test expected a discrepancy due to misuse of #pragma unroll, but outputs matched."

# Issue 2: Naming conflict with "stride"
# Use convolution strides different from 1 to expose potential confusion.
def test_stride_naming_conflict():
    cuda_module = build_kernel()
    N, C_in, H_in, W_in = 2, 4, 10, 10
    C_out = 5
    kH, kW = 3, 3
    stride = (2, 2)  # non-unit strides
    padding = (1, 1)

    x = torch.randn(N, C_in, H_in, W_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_in, C_out, kH, kW, device="cuda", dtype=torch.float32)
    bias = None

    out_kernel = cuda_module.forward(x, weight, bias, list(stride), list(padding))
    out_ref = reference_conv_transpose2d(x, weight, bias, stride, padding)
    # The misnamed local variable could be hidden now;
    # if the kernel logic is misinterpreting the grid stride vs convolution stride then the result will differ.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-4), \
        "Test expected a discrepancy due to naming conflict with 'stride', but outputs matched."

# Issue 3: Lack of error checking in the kernel launch
# We simulate an error by deliberately providing an input that is very likely to cause an out-of-bound access.
def test_error_checking():
    cuda_module = build_kernel()
    # Provide an input size and parameters that make the output index space very small,
    # then trigger an invalid memory access by mis-specifying padding.
    N, C_in, H_in, W_in = 1, 4, 5, 5
    C_out = 4
    kH, kW = 7, 7  # Large kernel relative to input, may force invalid accesses.
    stride = (1, 1)
    padding = (0, 0)  # Without padding the index math in the kernel might go out-of-bound.

    x = torch.randn(N, C_in, H_in, W_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_in, C_out, kH, kW, device="cuda", dtype=torch.float32)
    bias = None

    # The following call should trigger an error (or produce garbage) if error checking were in place.
    out_kernel = cuda_module.forward(x, weight, bias, list(stride), list(padding))
    # Force device synchronization to catch any asynchronous CUDA errors.
    torch.cuda.synchronize()
    # In a correct implementation an error would be detected;
    # here we check that the output does not match the PyTorch fallback.
    out_ref = reference_conv_transpose2d(x, weight, bias, stride, padding)
    assert not torch.allclose(out_kernel, out_ref, atol=1e-3), \
        "Test expected kernel error due to lack of error checking, but outputs matched."

# Issue 4: Data type support limited to float32.
def test_dtype_support():
    cuda_module = build_kernel()
    N, C_in, H_in, W_in = 2, 4, 8, 8
    C_out = 3
    kH, kW = 3, 3
    stride = (1, 1)
    padding = (1, 1)

    # Create double tensors (float64) to trigger dtype issue.
    x = torch.randn(N, C_in, H_in, W_in, device="cuda", dtype=torch.float64)
    weight = torch.randn(C_in, C_out, kH, kW, device="cuda", dtype=torch.float64)
    bias = None

    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(x, weight, bias, list(stride), list(padding))

# Issue 5: Assumption of contiguous memory.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    N, C_in, H_in, W_in = 2, 4, 10, 10
    C_out = 3
    kH, kW = 3, 3
    stride = (1, 1)
    padding = (1, 1)

    x = torch.randn(N, C_in, H_in, W_in, device="cuda", dtype=torch.float32)
    # Make x non-contiguous by transposing dimensions.
    x_nc = x.transpose(2, 3)
    weight = torch.randn(C_in, C_out, kH, kW, device="cuda", dtype=torch.float32)
    bias = None

    out_kernel = cuda_module.forward(x_nc, weight, bias, list(stride), list(padding))
    out_ref = reference_conv_transpose2d(x_nc.contiguous(), weight, bias, stride, padding)
    # The kernel expects contiguous inputs so the results may be off.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-4), \
        "Test expected discrepancy due to non-contiguous input but outputs matched."

# Issue 6: Inflexible constant memory usage for weight.
# We force a scenario where the weight tensor dimensions are unusual yet under the byte limit.
def test_constant_memory_edge():
    cuda_module = build_kernel()
    # Create a weight that is small in byte size but with an unusual layout.
    # For example, a weight tensor with non-standard dimensions but total size under 64KB.
    N, C_in, H_in, W_in = 1, 2, 5, 5
    C_out = 2
    kH, kW = 2, 3  # total elements = 2*2*2*3 = 24 floats
    stride = (1, 1)
    padding = (0, 0)

    x = torch.randn(N, C_in, H_in, W_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_in, C_out, kH, kW, device="cuda", dtype=torch.float32)
    bias = None

    out_kernel = cuda_module.forward(x, weight, bias, list(stride), list(padding))
    out_ref = reference_conv_transpose2d(x, weight, bias, stride, padding)
    # Depending on how the weight is laid out in constant memory the results might be off.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-4), \
        "Test expected discrepancy due to inflexible constant memory usage but outputs matched."
        
if __name__ == "__main__":
    pytest.main([__file__])
