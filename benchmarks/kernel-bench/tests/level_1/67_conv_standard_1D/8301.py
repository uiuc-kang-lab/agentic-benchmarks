
import torch
import torch.nn.functional as F
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

# Helper: perform 1D convolution using PyTorch's own implementation
def torch_conv1d(x, weight, bias, stride, padding, dilation, groups):
    return F.conv1d(x, weight, bias, stride, padding, dilation, groups)

# Test 1: Trigger type error by passing a tensor with the wrong dtype (double)
def test_input_tensor_wrong_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cuda_module = build_kernel()
    
    # Create double precision inputs (wrong type)
    N, C_in, L_in = 4, 3, 50
    C_out, K = 8, 3
    x = torch.randn(N, C_in, L_in, dtype=torch.float64, device='cuda')
    weight = torch.randn(C_out, C_in, K, dtype=torch.float64, device='cuda')
    bias = torch.randn(C_out, dtype=torch.float64, device='cuda')
    
    stride, padding, dilation, groups = 1, 1, 1, 1

    # Expect the kernel to throw an error because it checks for float32.
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)

# Test 2: Trigger non-contiguous input issue.
def test_non_contiguous_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cuda_module = build_kernel()
    
    # Create contiguous input tensors first.
    N, C_in, L_in = 4, 3, 50
    C_out, K = 8, 3
    stride, padding, dilation, groups = 1, 1, 1, 1

    # PyTorch nn.Conv1d weight layout is (C_out, C_in/groups, kernel_size)
    # Here groups == 1 so weight shape is (C_out, C_in, K)
    x = torch.randn(N, C_in, L_in, device='cuda', dtype=torch.float32)
    weight = torch.randn(C_out, C_in, K, device='cuda', dtype=torch.float32)
    bias = torch.randn(C_out, device='cuda', dtype=torch.float32)

    # Make x non-contiguous without changing its shape:
    # Permute and then re-permute in a way that does not call .contiguous()
    x_noncontig = x.transpose(1,2).transpose(1,2)
    assert not x_noncontig.is_contiguous(), "The test tensor should be non-contiguous"

    # Compute using our custom CUDA kernel
    y_kernel = cuda_module.forward(x_noncontig, weight, bias, stride, padding, dilation, groups)
    # Compute reference result using PyTorch conv1d on the contiguous tensor
    y_ref = torch_conv1d(x, weight, bias, stride, padding, dilation, groups)
    
    # In case of non-contiguous input, the kernel (which assumes contiguous layout) may compute incorrect results.
    # We require that the outputs differ.
    if torch.allclose(y_kernel, y_ref, atol=1e-5):
        pytest.fail("Kernel output matches reference output on non-contiguous input; expected a discrepancy due to lack of contiguous check.")

