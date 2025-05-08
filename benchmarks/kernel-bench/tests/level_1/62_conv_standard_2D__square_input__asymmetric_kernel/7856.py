
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="conv2d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_non_unity_dilation_groups():
    # Issue 1: The kernel ignores non-unity dilation and groups.
    # When dilation != 1 or groups != 1, the forward() function falls back to torch::conv2d.
    module = build_kernel()
    x = torch.randn(1, 3, 10, 10, device="cuda", dtype=torch.float32)
    weight = torch.randn(64, 3, 3, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(64, device="cuda", dtype=torch.float32)
    
    # Here, setting dilation != 1 forces the fallback.
    output = module.forward(x, weight, bias, stride=1, padding=1, dilation=2, groups=1)
    expected = F.conv2d(x, weight, bias, stride=1, padding=1, dilation=2, groups=1)
    assert torch.allclose(output, expected, atol=1e-5), \
        "Fallback for non-unity dilation did not produce expected results."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_unused_boundary_computations():
    # Issue 2: The kernel computes h_start/h_end and w_start/w_end but does not use them.
    # While this may not change correctness, it can lead to subtle bugs at boundary conditions.
    # We test on inputs where some kernel windows would extend past boundaries.
    module = build_kernel()
    x = torch.randn(2, 3, 8, 8, device="cuda", dtype=torch.float32)
    weight = torch.randn(4, 3, 5, 5, device="cuda", dtype=torch.float32)  # Kernel larger than input spatial dims
    bias = None
    
    output = module.forward(x, weight, bias, stride=1, padding=2, dilation=1, groups=1)
    expected = F.conv2d(x, weight, bias, stride=1, padding=2, dilation=1, groups=1)
    assert torch.allclose(output, expected, atol=1e-5), \
        "Kernel boundary handling appears incorrect when kernel windows extend past input boundaries."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_float32_only_support():
    # Issue 3: The kernel is hard-coded for float32.
    # Passing half precision (float16) tensors will lead to incorrect behavior because the kernel calls
    # data_ptr<float>() regardless of the tensor dtype.
    module = build_kernel()
    x = torch.randn(1, 3, 10, 10, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, 3, 3, 3, device="cuda", dtype=torch.float16)
    bias = None

    # Compute expected result using a conversion to float32 then back to float16.
    expected = F.conv2d(x.float(), weight.float(), bias, stride=1, padding=1).half()
    output = module.forward(x, weight, bias, stride=1, padding=1, dilation=1, groups=1)
    # We expect the kernel to miscompute because it treats the data as float32.
    with pytest.raises(AssertionError):
        assert torch.allclose(output, expected, atol=1e-3), \
            "Kernel unexpectedly handled float16 inputs correctly, but it is hard-coded for float32."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_asymmetric_stride_padding():
    # Issue 4: The kernel and its wrapper assume symmetric (scalar) stride and padding parameters.
    # If tuple values are provided (i.e. asymmetric), the C++ extension will likely fail during argument parsing.
    module = build_kernel()
    x = torch.randn(1, 3, 10, 10, device="cuda", dtype=torch.float32)
    weight = torch.randn(64, 3, 3, 5, device="cuda", dtype=torch.float32)  # Asymmetric kernel dimensions
    bias = None

    with pytest.raises(TypeError):
        module.forward(x, weight, bias, stride=(1, 1), padding=(1, 2), dilation=1, groups=1)
