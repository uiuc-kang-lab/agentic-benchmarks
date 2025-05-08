
import pytest
import torch
from torch import nn
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

# Helper: create a reference depthwise separable convolution using PyTorch built-in modules.
class RefModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_contiguous_input():
    # Issue 1: non-contiguous tensors
    batch_size, in_channels, height, width = 4, 3, 32, 32
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    # create a model and get its parameters
    ref_model = RefModel(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, bias=True).cuda()
    # get weights and biases for depthwise and pointwise
    depthwise_weight = ref_model.depthwise.weight.data.clone()
    pointwise_weight = ref_model.pointwise.weight.data.clone()
    depthwise_bias = ref_model.depthwise.bias.data.clone() if ref_model.depthwise.bias is not None else torch.tensor([])
    pointwise_bias = ref_model.pointwise.bias.data.clone() if ref_model.pointwise.bias is not None else torch.tensor([])

    # Create a contiguous input and then make it non-contiguous by transposing
    x = torch.randn(batch_size, in_channels, height, width, device="cuda")
    x_noncontig = x.transpose(1, 2)  # now shape (B, H, C, W) which is non-standard and non-contiguous

    kernel_mod = build_kernel()

    # Call our custom kernel: assuming wrapper signature
    with pytest.raises(AssertionError) as excinfo:
        # The custom kernel uses manual arithmetic that assumes contiguous NCHW layout.
        # In our test, we expect the output not to match the reference.
        out_custom = kernel_mod.forward(
            x_noncontig, 
            depthwise_weight, 
            pointwise_weight, 
            depthwise_bias if depthwise_bias.numel() else torch.tensor(), 
            pointwise_bias if pointwise_bias.numel() else torch.tensor(), 
            stride, padding, dilation
        )
    # If the kernel did not crash, then likely the output is wrong.
    # In that case, we compute reference on contiguous input and compare.
    # (Here we simply check that the results differ.)
    if not excinfo.value:
        x_contig = x_noncontig.contiguous()  # force a re-layout
        out_contig = ref_model(x_contig)
        out_custom = kernel_mod.forward(
            x_noncontig, 
            depthwise_weight, 
            pointwise_weight, 
            depthwise_bias if depthwise_bias.numel() else torch.tensor(), 
            pointwise_bias if pointwise_bias.numel() else torch.tensor(), 
            stride, padding, dilation
        )        
        # They should not be close due to wrong indexing.
        assert not torch.allclose(out_custom, out_contig, atol=1e-5), \
            "Non-contiguous input should trigger incorrect indexing, but outputs match."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_invalid_output_dimensions():
    # Issue 5: the kernel does not guard against output dims <= 0.
    # Choose parameters that yield negative output height: in_h too small relative to kernel_size.
    batch_size, in_channels, height, width = 2, 3, 2, 2  # height=2, width=2 will be too small for kernel_size=3
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    
    x = torch.randn(batch_size, in_channels, height, width, device="cuda")
    # Create dummy weights of valid shapes:
    depthwise_weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda")
    pointwise_weight = torch.randn(out_channels, in_channels, 1, 1, device="cuda")
    depthwise_bias = torch.randn(in_channels, device="cuda")
    pointwise_bias = torch.randn(out_channels, device="cuda")
    
    kernel_mod = build_kernel()
    # Expect the kernel launch to fail (or produce error) because out_h and out_w become non-positive.
    with pytest.raises(RuntimeError):
        kernel_mod.forward(
            x, depthwise_weight, pointwise_weight,
            depthwise_bias, pointwise_bias,
            stride, padding, dilation
        )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_runtime_dependent_unroll_issue():
    # Issue 3: Using non-constant kernel sizes might mis-optimize the loop unrolling.
    # Here we choose kernel_size=5 to see if the kernel output is (numerically) wrong.
    # (In a real scenario, miscompilation might yield wrong results.)
    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels = 6
    kernel_size = 5
    stride = 1
    padding = 2  # with padding=2, output size remains same as input
    dilation = 1

    ref_model = RefModel(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, bias=True).cuda()
    depthwise_weight = ref_model.depthwise.weight.data.clone()
    pointwise_weight = ref_model.pointwise.weight.data.clone()
    depthwise_bias = ref_model.depthwise.bias.data.clone() if ref_model.depthwise.bias is not None else torch.tensor([])
    pointwise_bias = ref_model.pointwise.bias.data.clone() if ref_model.pointwise.bias is not None else torch.tensor([])

    x = torch.randn(batch_size, in_channels, height, width, device="cuda")

    kernel_mod = build_kernel()
    out_custom = kernel_mod.forward(
        x, depthwise_weight, pointwise_weight,
        depthwise_bias if depthwise_bias.numel() else torch.tensor(),
        pointwise_bias if pointwise_bias.numel() else torch.tensor(),
        stride, padding, dilation
    )
    out_ref = ref_model(x)
    # In case of issues with unrolling, the numerical results would differ.
    # We test that they are not matching within a tight tolerance.
    assert not torch.allclose(out_custom, out_ref, atol=1e-5), \
        "For kernel_size > 3, the custom kernel unexpectedly produced results matching the reference."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_tensor_input_error():
    # Issue 4: The kernel checks that the input tensor is on CUDA.
    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, height, width, device="cpu")
    depthwise_weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cpu")
    pointwise_weight = torch.randn(out_channels, in_channels, 1, 1, device="cpu")
    depthwise_bias = torch.randn(in_channels, device="cpu")
    pointwise_bias = torch.randn(out_channels, device="cpu")

    kernel_mod = build_kernel()
    with pytest.raises(RuntimeError):
        kernel_mod.forward(
            x, depthwise_weight, pointwise_weight,
            depthwise_bias, pointwise_bias,
            stride, padding, dilation
        )
