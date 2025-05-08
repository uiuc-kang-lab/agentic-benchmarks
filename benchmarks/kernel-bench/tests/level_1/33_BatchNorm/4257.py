
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="batchnorm_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper function to mimic BatchNorm2d using PyTorch (only for float inputs)
def pytorch_batchnorm_forward(input, weight, bias, running_mean, running_var, training, momentum, eps):
    bn = torch.nn.BatchNorm2d(input.size(1), momentum=momentum, eps=eps)
    # set the parameters manually for fair comparison
    bn.weight.data = weight.clone()
    bn.bias.data = bias.clone()
    bn.running_mean.data = running_mean.clone()
    bn.running_var.data = running_var.clone()
    bn.train(training)
    return bn(input)

# Issue 1: Data type incompatibility – kernel only supports float32.
def test_dtype_issue():
    cuda_module = build_kernel()
    
    # create double precision input (incorrect type for the kernel)
    N, C, H, W = 4, 3, 8, 8
    input_double = torch.randn(N, C, H, W, dtype=torch.double, device='cuda')
    weight_double = torch.randn(C, dtype=torch.double, device='cuda')
    bias_double = torch.randn(C, dtype=torch.double, device='cuda')
    running_mean_double = torch.zeros(C, dtype=torch.double, device='cuda')
    running_var_double = torch.ones(C, dtype=torch.double, device='cuda')
    
    # Invoke the kernel that expects float32 but get double input.
    # The kernel does not check the data type and misinterprets the memory.
    out = cuda_module.forward(
        input_double, weight_double, bias_double,
        running_mean_double, running_var_double,
        True, 0.1, 1e-5
    )
    torch.cuda.synchronize()

    # Compute a reference output using PyTorch's BatchNorm2d on a float tensor.
    input_float = input_double.float()
    weight_float = weight_double.float()
    bias_float = bias_double.float()
    running_mean_float = running_mean_double.float()
    running_var_float = running_var_double.float()

    ref_out = pytorch_batchnorm_forward(input_float, weight_float, bias_float,
                                        running_mean_float, running_var_float,
                                        True, 0.1, 1e-5)
    # Because of the type reinterpretation, the kernel output should differ significantly.
    # We use a loose tolerance to show that the output is not matching.
    assert not torch.allclose(out, ref_out, atol=1e-2), \
        "Kernel incorrectly accepted a double tensor, which it should not."

# Issue 2: Input dimension assumption – kernel assumes a 4D tensor.
def test_dimension_issue():
    cuda_module = build_kernel()
    
    # Create a 3D input tensor instead of the expected 4D tensor.
    N, C, H = 4, 3, 8  # missing the fourth dimension (W)
    input_wrong_dim = torch.randn(N, C, H, device='cuda', dtype=torch.float32)
    weight = torch.randn(C, dtype=torch.float32, device='cuda')
    bias = torch.randn(C, dtype=torch.float32, device='cuda')
    running_mean = torch.zeros(C, dtype=torch.float32, device='cuda')
    running_var = torch.ones(C, dtype=torch.float32, device='cuda')
    
    # The kernel code uses input.size(3) and will throw an error or produce out-of-bound access.
    with pytest.raises(IndexError):
        _ = cuda_module.forward(
            input_wrong_dim, weight, bias,
            running_mean, running_var,
            True, 0.1, 1e-5
        )
