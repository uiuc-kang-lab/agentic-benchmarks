
import os
import re
import tempfile
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to build a kernel module from a source file.
def build_kernel(source_path, extra_cuda_cflags=None):
    module = load(
        name="test_module",
        sources=[source_path],
        extra_cuda_cflags=extra_cuda_cflags if extra_cuda_cflags is not None else ["-O3", "--use_fast_math"],
        verbose=False,
    )
    return module

# 1. Test for input type issue: pass half precision and check for failure.
def test_input_tensor_type():
    # Create a simple InstanceNorm input tensor in half precision.
    N, C, H, W = 2, 4, 16, 16
    x_half = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
    weight = torch.randn(C, device="cuda", dtype=torch.float16)
    bias = torch.randn(C, device="cuda", dtype=torch.float16)
    
    # Build the kernel normally.
    kernel_module = build_kernel("kernel.cu")
    
    # We expect that using half tensors (instead of float32) will lead to an error,
    # because the kernel explicitly casts pointers to float *.
    with pytest.raises(Exception):
        y = kernel_module.forward(x_half, weight, bias, 1e-5)
        torch.cuda.synchronize()

# 2. Test for partial warp reduction issue:
# We simulate this problem by modifying the kernel source so that the block size is not a multiple of 32.
def test_partial_warp_block_size():
    # Create a temporary copy of the kernel file with modified block_size.
    with open("kernel.cu", "r") as f:
        source = f.read()
    
    # Patch the line that defines the block size.
    # Original line:   int block_size = 256; 
    # We change it to a value not divisible by 32, say 250.
    patched_source = re.sub(r"int block_size = 256;", "int block_size = 250;", source)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False) as tmp:
        tmp.write(patched_source)
        tmp_path = tmp.name

    try:
        kernel_module = build_kernel(tmp_path)
        N, C, H, W = 2, 8, 16, 16
        # Use float32 input.
        x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
        # Create weight and bias (affine parameters) 
        weight = torch.randn(C, device="cuda", dtype=torch.float32)
        bias = torch.randn(C, device="cuda", dtype=torch.float32)
        
        y_kernel = kernel_module.forward(x, weight, bias, 1e-5)
        torch.cuda.synchronize()

        # Compare against PyTorch’s implementation.
        # Create an InstanceNorm2d module (without running running stats).
        m = torch.nn.InstanceNorm2d(num_features=C, eps=1e-5, affine=True)
        # Force the module’s parameters to match our weight and bias.
        with torch.no_grad():
            m.weight.copy_(weight)
            m.bias.copy_(bias)
        y_ref = m(x)
        # Due to reduction bug the output may be noticeably off.
        assert not torch.allclose(y_kernel, y_ref, atol=1e-3), "Kernel output unexpectedly matches reference output even with partial warp bug."
    finally:
        os.remove(tmp_path)

# 3. Test for large shared memory over–allocation:
def test_large_shared_memory():
    # Use very large H and W to force shared memory allocation to be huge.
    N, C, H, W = 2, 4, 1024, 1024  # This request ~ 1024*1024*4 bytes = ~4 MB per block.
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    
    kernel_module = build_kernel("kernel.cu")
    
    # This should trigger a CUDA kernel launch error (or implicit error) because the allocated shared memory exceeds the limit.
    with pytest.raises(RuntimeError):
        y = kernel_module.forward(x, weight, bias, 1e-5)
        torch.cuda.synchronize()

# 4. Test for non–contiguous input:
def test_non_contiguous_input():
    N, C, H, W = 2, 4, 16, 16
    # Create a contiguous tensor and then make a non–contiguous view (e.g. by transposing)
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    x_noncontig = x.transpose(1,2)  # Now tensor shape becomes (N, H, C, W) which is non–contiguous for the intended layout.
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    
    kernel_module = build_kernel("kernel.cu")
    
    # Because the kernel expects the input to be laid out as (N, C, H, W), using a non–contiguous tensor
    # will lead to a misinterpretation of the memory layout. We expect that the resulting output is incorrect.
    y_kernel = kernel_module.forward(x_noncontig, weight, bias, 1e-5)
    torch.cuda.synchronize()
    
    # We compute a reference result using InstanceNorm2d after forcing the tensor to be contiguous.
    m = torch.nn.InstanceNorm2d(num_features=C, eps=1e-5, affine=True)
    with torch.no_grad():
        m.weight.copy_(weight)
        m.bias.copy_(bias)
    # The reference input is forced contiguous and with correct layout
    y_ref = m(x.contiguous())
    
    # We expect that the kernel output is different from the reference (i.e. it is wrong).
    assert not torch.allclose(y_kernel, y_ref, atol=1e-4), "Kernel output unexpectedly matches reference output even with non-contiguous input."

if __name__ == '__main__':
    pytest.main([__file__])
