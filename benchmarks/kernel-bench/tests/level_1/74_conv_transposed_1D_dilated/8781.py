
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="conv_transpose1d_cuda",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Data type limitation (only supports float32)
def test_dtype_issue():
    cuda_module = build_kernel()
    # Create input tensors in double precision. The kernel expects float32.
    N = 2
    C_in = 3
    L_in = 16
    C_out = 4
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    x = torch.randn(N, C_in, L_in, device="cuda", dtype=torch.double)
    weight = torch.randn(C_in, C_out, kernel_size, device="cuda", dtype=torch.double)
    # The kernel binding calls .data_ptr<float>(), so this should lead to wrong results.
    with pytest.raises(RuntimeError):
        # Expect that wrong dtype leads to an error or misbehaviour.
        y = cuda_module.forward(x, weight, None, stride, padding, dilation)
        # Force a synchronization to capture any asynchronous CUDA errors.
        torch.cuda.synchronize()

# Issue 2: Lack of output_padding support.
def test_no_output_padding():
    # In PyTorch's ConvTranspose1d, one can set output_padding to adjust the output size.
    # Here, the kernel does not expose output_padding so it will compute L_out as:
    #   L_out = (L_in - 1)*stride - 2*padding + dilation*(kernel_size - 1) + 1
    # We purposely choose parameters that in PyTorch with output_padding would produce a different shape.
    cuda_module = build_kernel()
    N = 1
    C_in = 2
    L_in = 10
    C_out = 3
    kernel_size = 4
    stride = 2
    padding = 1
    dilation = 1
    # Expected L_out with an output_padding parameter (say, output_padding=1) would be:
    #   L_out_expected = (L_in - 1)*stride - 2*padding + dilation*(kernel_size - 1) + 1 + 1
    # Since the kernel does not support output_padding, the output length will be:
    #   L_out_kernel = (L_in - 1)*stride - 2*padding + dilation*(kernel_size - 1) + 1
    x = torch.randn(N, C_in, L_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_in, C_out, kernel_size, device="cuda", dtype=torch.float32)
    y = cuda_module.forward(x, weight, None, stride, padding, dilation)
    # Compute L_out as per the kernel:
    L_out_kernel = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    # Simulate a scenario where the caller expected an extra output pixel.
    L_out_expected = L_out_kernel + 1
    assert y.size(2) != L_out_expected, (
        "Kernel does not support output_padding but produced an output shape matching "
        "the shape expected when output_padding is used."
    )

# Issue 3: Assumed weight memory layout.
def test_weight_layout_issue():
    cuda_module = build_kernel()
    N = 1
    C_in = 3
    L_in = 20
    C_out = 5
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    x = torch.randn(N, C_in, L_in, device="cuda", dtype=torch.float32)
    # Instead of the expected layout [C_in, C_out, kernel_size],
    # we deliberately create a weight tensor with a common alternative layout [C_out, C_in, kernel_size].
    # This reversed layout is not supported by our kernel.
    weight_wrong = torch.randn(C_out, C_in, kernel_size, device="cuda", dtype=torch.float32)
    # The kernel expects a contiguous tensor in the layout [C_in, C_out, kernel_size]. So the result will be wrong.
    y = cuda_module.forward(x, weight_wrong, None, stride, padding, dilation)
    # For a simple test, we can check that the output is not equal to an expected reference computed
    # using intuitive transposed convolution from PyTorch’s native implementation.
    conv_transpose1d = torch.nn.ConvTranspose1d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    ).eval().cuda()
    # Manually set the weight with the correctly transposed version of weight_wrong.
    # Since our kernel assumes weight shape [C_in, C_out, kernel_size],
    # we set the weight of the reference module as the transpose of weight_wrong along dims 0 and 1.
    conv_transpose1d.weight.data = weight_wrong.transpose(0, 1).contiguous()
    y_ref = conv_transpose1d(x)
    # The outputs should differ because the kernel used the wrong layout.
    with pytest.raises(AssertionError):
        assert torch.allclose(y, y_ref, atol=1e-4), (
            "Kernel unexpectedly produced an output matching the reference "
            "despite an incorrect weight layout."
        )
