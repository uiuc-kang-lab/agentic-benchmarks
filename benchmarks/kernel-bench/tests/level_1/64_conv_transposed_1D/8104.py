
import pytest
import torch
from torch import nn
from torch.utils.cpp_extension import load
import os

# Ensure that we compile the extension from kernel.cu in the current directory.
# The module name is "test_module"
@pytest.fixture(scope="session")
def cuda_extension():
    # Build and load the CUDA extension
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# A dummy model that calls the CUDA kernel via the extension.
# We mimic the behavior in the original PyTorch code.
class ConvTranspose1dCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding, groups):
        # Call the CUDA forward function from our extension.
        return torch.ops.test_module.forward(input, weight, bias, int(stride), int(padding), int(output_padding), int(groups))
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward is not implemented")

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Create weight and, if required, bias parameters.
        # The weight shape is assumed to be (in_channels, out_channels/groups, kernel_size)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, kernel_size, device="cuda", dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device="cuda", dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x):
        return ConvTranspose1dCudaFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups
        )

# Test case to trigger issue #1:
# We provide an input where the work per block (total_iters) is very small compared to the fixed
# 128-thread block configuration. This may expose the underlying assumption about exactly four warps
# in the reduction buffer (which is hard-coded to size 4).
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fixed_shared_memory_reduction(cuda_extension):
    # Use a configuration with minimal kernel_size and in_channels so that total_iters is small.
    batch_size = 2
    in_channels = 4
    out_channels = 2
    kernel_size = 1
    length = 8
    stride = 2
    padding = 0
    output_padding = 0
    groups = 1

    # Create the model; it uses float32 parameters and the kernel expects 128 threads per block,
    # but with so little work the intra-block reduction will underutilize threads and may read
    # uninitialized shared memory if the kernel is not generalized.
    model = Model(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias=True).cuda()

    # Create a contiguous input tensor
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    try:
        # Run the forward pass; if there is a shared memory bug, the result may be incorrect.
        y = model(x)
    except Exception as e:
        pytest.fail("Kernel crashed with exception: " + str(e))
    
    # Perform a check that compares against PyTorch's own ConvTranspose1d
    conv_transpose = torch.nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding,
        output_padding=output_padding, groups=groups, bias=True
    ).cuda()
    # Force conv_transpose to have the same parameters as our model.
    conv_transpose.weight.data.copy_(model.weight.data)
    if model.bias is not None:
        conv_transpose.bias.data.copy_(model.bias.data)
    
    y_ref = conv_transpose(x)
    # This test is intended to expose reduction bugs, so if the results differ significantly, we fail.
    assert torch.allclose(y, y_ref, atol=1e-4), f"Output mismatch possibly due to fixed shared memory size issue. Max difference: {(y - y_ref).abs().max().item()}"

# Test case to trigger issue #2:
# We pass input and weight tensors of type float64. Since the kernel only handles float32,
# the computed results will be incorrect.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_float_type_mismatch(cuda_extension):
    batch_size = 2
    in_channels = 8
    out_channels = 4
    kernel_size = 3
    length = 16
    stride = 1
    padding = 1
    output_padding = 0
    groups = 1

    # Create a model with parameters cast to float64.
    # Note: We deliberately use float64, which is not supported by the kernel.
    model = Model(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias=True).cuda()
    model.weight.data = model.weight.data.double()
    if model.bias is not None:
        model.bias.data = model.bias.data.double()
    
    # Prepare an input tensor in float64
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float64)
    # The kernel will reinterpret the underlying memory as float32 and produce wrong results.
    y = model(x)

    # Use PyTorch's built-in ConvTranspose1d (which supports float64) as the reference
    conv_transpose = torch.nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding,
        output_padding=output_padding, groups=groups, bias=True
    ).cuda().double()
    conv_transpose.weight.data.copy_(model.weight.data)
    if model.bias is not None:
        conv_transpose.bias.data.copy_(model.bias.data)
    y_ref = conv_transpose(x)

    # We expect the outputs not to match due to type mismatch.
    with pytest.raises(AssertionError):
        assert torch.allclose(y, y_ref, atol=1e-4), "Kernel produced correct results despite type mismatch (expected error due to float64 input)."
