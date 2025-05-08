
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel(extra_cuda_cflags=None):
    cuda_module = load(
        name="test_batchnorm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags if extra_cuda_cflags is not None else ["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper: reference batchnorm using PyTorch
class RefBatchNorm(torch.nn.Module):
    def __init__(self, num_features, momentum, eps):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(num_features=num_features, momentum=momentum, eps=eps)
        # Force evaluation mode in order to use running statistics:
        self.bn.train()

    def forward(self, x):
        return self.bn(x)

# Test case 1: Input tensor type is not float32.
def test_input_tensor_dtype():
    # Create input of type float64 (double)
    batch_size, features, dim1, dim2 = 16, 64, 256, 256
    x = torch.randn(batch_size, features, dim1, dim2, dtype=torch.float64, device='cuda')
    weight = torch.randn(features, device='cuda', dtype=torch.float64)
    bias = torch.randn(features, device='cuda', dtype=torch.float64)
    running_mean = torch.zeros(features, device='cuda', dtype=torch.float64)
    running_var = torch.ones(features, device='cuda', dtype=torch.float64)
    
    module = build_kernel()

    with pytest.raises(RuntimeError):
        # The kernel expects float32 but is given float64, so it should error out (or produce a CUDA runtime error).
        _ = module.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
        

# Test case 2: Input tensor is not 4D.
def test_input_tensor_dimension():
    # Create a 3D tensor (e.g., missing one spatial dimension).
    batch_size, features, dim1 = 16, 64, 256
    x = torch.randn(batch_size, features, dim1, device='cuda', dtype=torch.float32)
    weight = torch.randn(features, device='cuda', dtype=torch.float32)
    bias = torch.randn(features, device='cuda', dtype=torch.float32)
    running_mean = torch.zeros(features, device='cuda', dtype=torch.float32)
    running_var = torch.ones(features, device='cuda', dtype=torch.float32)
    
    module = build_kernel()

    with pytest.raises(IndexError):
        # The kernel's index arithmetic assumes a 4D tensor.
        _ = module.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
        

# Test case 3: Input tensor is non-contiguous.
def test_non_contiguous_input():
    # Create a contiguous tensor and then create a non-contiguous view.
    batch_size, features, dim1, dim2 = 16, 64, 256, 256
    x_full = torch.randn(batch_size, features, dim1, dim2, device='cuda', dtype=torch.float32)
    # Slicing will usually preserve contiguity but transposing will break it.
    x_non_contig = x_full.transpose(1, 2)  # Now shape is (batch_size, dim1, features, dim2) and non-contiguous.
    
    weight = torch.randn(features, device='cuda', dtype=torch.float32)
    bias = torch.randn(features, device='cuda', dtype=torch.float32)
    running_mean = torch.zeros(features, device='cuda', dtype=torch.float32)
    running_var = torch.ones(features, device='cuda', dtype=torch.float32)
    
    module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # The kernel launch code calls the CHECK_CONTIGUOUS macro. A non-contiguous input should trigger an error.
        _ = module.forward(x_non_contig, weight, bias, running_mean, running_var, True, 0.1, 1e-5)


# Test case 4: Misaligned memory for weight and bias tensors.
def test_misaligned_weight_bias():
    # Create a tensor that is contiguous but intentionally misaligned by slicing off the first element.
    batch_size, features, dim1, dim2 = 16, 64, 256, 256
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda', dtype=torch.float32)
    # Create a larger tensor and take a slice starting at index 1 to force a misaligned pointer.
    weight_full = torch.randn(features + 1, device='cuda', dtype=torch.float32)
    bias_full = torch.randn(features + 1, device='cuda', dtype=torch.float32)
    # Slice off the first element; the resulting tensors are contiguous but may have an internal offset.
    weight = weight_full.narrow(0, 1, features)
    bias = bias_full.narrow(0, 1, features)
    running_mean = torch.zeros(features, device='cuda', dtype=torch.float32)
    running_var = torch.ones(features, device='cuda', dtype=torch.float32)
    
    module = build_kernel()

    # Run the kernel and compare to PyTorch's BatchNorm output.
    # Because the kernel assumes 128-bit alignment for weight/bias, the misalignment may lead to
    # an output that significantly deviates from the expected value.
    output_kernel = module.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
    
    # For reference, compute BatchNorm using PyTorch.
    ref_bn = RefBatchNorm(features, 0.1, 1e-5).to('cuda')
    # We set weight and bias to the same values as used in the kernel.
    with torch.no_grad():
        ref_bn.bn.weight.copy_(weight)
        ref_bn.bn.bias.copy_(bias)
        # Manually set the running_mean and running_var to the same values.
        ref_bn.bn.running_mean.copy_(running_mean)
        ref_bn.bn.running_var.copy_(running_var)

    output_ref = ref_bn(x)
    
    max_diff = (output_kernel - output_ref).abs().max().item()
    # This test should fail (i.e. max_diff above tolerance) if the kernel is adversely affected by misaligned weight/bias.
    assert max_diff > 1e-3, f"Kernel output appears unaffected by misaligned weight/bias (max diff: {max_diff})"

