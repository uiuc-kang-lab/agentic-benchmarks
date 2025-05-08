
import os
import tempfile
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility: load the CUDA extension from "kernel.cu" (as shipped)
def build_default_module():
    module = load(
        name="default_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return module

# Utility: load a modified CUDA extension with a custom thread count.
def build_module_with_non_power2_threads(custom_threads=250):
    # Read original source
    with open("kernel.cu", "r") as f:
        source = f.read()
    # Replace the hardcoded threads (256 => custom_threads)
    # We assume the forward_cuda function sets:
    #    const int threads = 256;
    # Replace that with:
    #    const int threads = <custom_threads>;
    new_source = source.replace("const int threads = 256;", f"const int threads = {custom_threads};")
    # Write to temporary file
    temp_dir = tempfile.mkdtemp()
    temp_filename = os.path.join(temp_dir, "kernel_custom.cu")
    with open(temp_filename, "w") as f:
        f.write(new_source)
    module = load(
        name="custom_module",
        sources=[temp_filename],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return module

# Issue 1: Constant memory limitation (channels > 1024 hit constant memory bounds)
def test_constant_memory_limit_exceed():
    # Use a channel count larger than 1024.
    N = 4
    C = 1025  # deliberately 1 above the constant memory size
    H, W = 16, 16
    # Create input of proper size.
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    # Also create weight and bias for BN
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    # Create running statistics tensors. Initialize with zeros, ones.
    running_mean = torch.zeros(C, device="cuda", dtype=torch.float32)
    running_var = torch.ones(C, device="cuda", dtype=torch.float32)
    
    module = build_default_module()
    # The kernel copies the weight and bias to fixed constant memory.
    # With C > 1024 this is an out‐of‐bounds access.
    # We expect that this may produce a wrong result or an illegal memory access.
    with pytest.raises(RuntimeError):
        # It is acceptable if the kernel crashes (raising RuntimeError on cuda error)
        res = module.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
        torch.cuda.synchronize()

# Issue 2: Reduction algorithm assumes power-of-two thread count.
def test_non_power_of_two_threads():
    # Build a version of the module that launches the kernel with 250 threads per block.
    module = build_module_with_non_power2_threads(custom_threads=250)
    N = 4
    C = 32  # use modest channel count
    H, W = 16, 16
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    running_mean = torch.zeros(C, device="cuda", dtype=torch.float32)
    running_var = torch.ones(C, device="cuda", dtype=torch.float32)
    
    # In training mode the kernel computes per-channel means with a block reduction.
    # Due to the non-power-of-two thread count the reduction might be computed incorrectly.
    out = module.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
    torch.cuda.synchronize()
    
    # We compare against PyTorch’s built-in batchnorm (in training mode).
    bn = torch.nn.BatchNorm2d(C, momentum=0.1, eps=1e-5).cuda()
    # Set bn weights to be the same as our CUDA module constant memory (simulate)
    with torch.no_grad():
        bn.weight.copy_(weight)
        bn.bias.copy_(bias)
        bn.running_mean.copy_(running_mean)
        bn.running_var.copy_(running_var)
    ref = bn(x)
    # Because of the bad reduction the result should not agree.
    # Here we assert that they are NOT close (the test passes when a noticeable error is observed)
    assert not torch.allclose(out, ref, atol=1e-3), "Reduction error was not triggered with non-power-of-2 thread count."

# Issue 3: Kernel only supports float32, but no explicit type check is done.
def test_non_float_input():
    N = 4
    C = 32
    H, W = 16, 16
    # Provide input in float64 instead of float32.
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float64)
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    running_mean = torch.zeros(C, device="cuda", dtype=torch.float32)
    running_var = torch.ones(C, device="cuda", dtype=torch.float32)
    
    module = build_default_module()
    # The checks only verify that the tensor is CUDA and contiguous. No data type check.
    # When we call the kernel, the input pointer is cast to float* and the data interpreted as float.
    # That will produce a wrong answer.
    out = module.forward(x, weight, bias, running_mean, running_var, False, 0.1, 1e-5)
    torch.cuda.synchronize()
    
    # We compute the reference batchnorm (forcing conversion of the input to float32)
    x32 = x.float()
    bn = torch.nn.BatchNorm2d(C, momentum=0.1, eps=1e-5).cuda()
    with torch.no_grad():
        bn.weight.copy_(weight)
        bn.bias.copy_(bias)
        bn.running_mean.copy_(running_mean)
        bn.running_var.copy_(running_var)
    ref = bn(x32)
    # Since the kernel interpreted the double data as float data, the results will be very different.
    assert not torch.allclose(out, ref, atol=1e-3), "Kernel unexpectedly produced a reasonable result for non-FP32 inputs."

# Additional check: non-contiguous tensor should be rejected.
def test_non_contiguous_input():
    N = 4
    C = 32
    H, W = 16, 16
    # Create a contiguous tensor and then transpose to make it non-contiguous.
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32).transpose(2, 3)
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    running_mean = torch.zeros(C, device="cuda", dtype=torch.float32)
    running_var = torch.ones(C, device="cuda", dtype=torch.float32)
    
    module = build_default_module()
    with pytest.raises(RuntimeError):
        module.forward(x, weight, bias, running_mean, running_var, False, 0.1, 1e-5)
        torch.cuda.synchronize()
