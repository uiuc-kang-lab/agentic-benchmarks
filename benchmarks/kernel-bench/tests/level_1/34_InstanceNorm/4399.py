
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="instance_norm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: The kernel only supports float32 inputs.
def test_input_dtype():
    cuda_module = build_kernel()
    # Create a double precision tensor
    N, C, H, W = 4, 3, 16, 16
    x = torch.randn(N, C, H, W, dtype=torch.float64, device="cuda")
    weight = torch.randn(C, dtype=torch.float64, device="cuda")
    bias = torch.randn(C, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError):
        # Expecting a type error (or a failure later on) because the kernel only handles float32
        cuda_module.forward(x, weight, bias, 1e-5)
    torch.cuda.synchronize()

# Issue 2: The kernel assumes contiguous input. Here, we provide a non-contiguous tensor.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    N, C, H, W = 4, 3, 16, 16
    # Create a contiguous tensor and then make it non-contiguous by transposing spatial dims
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32).transpose(2, 3)
    # weight and bias are contiguous and correct
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    # The kernel does not check or correct non-contiguous layout.
    # The result is likely to be incorrect even if no error is thrown.
    y = cuda_module.forward(x, weight, bias, 1e-5)
    # Instead of a direct allclose, we check that the output is not equal to that of a correct implementation.
    # Use the PyTorch built-in InstanceNorm2d as a reference.
    norm_ref = torch.nn.InstanceNorm2d(C, eps=1e-5).to(x.device)
    # Manually copy weight and bias into norm_ref parameters for fair comparison.
    with torch.no_grad():
        norm_ref.weight.copy_(weight)
        norm_ref.bias.copy_(bias)
    y_ref = norm_ref(x.contiguous())
    # Expect that the output differs from the reference.
    assert not torch.allclose(y, y_ref, atol=1e-3), "Kernel unexpectedly handled non-contiguous input correctly."

# Issue 3: The kernel does not account for the extra shared memory required by blockReduceSum.
# For some input sizes, the actual shared memory required might be higher than what is allocated.
# We simulate this by setting a spatial size that forces a large dynamic shared memory allocation.
def test_excessive_shared_memory_allocation():
    cuda_module = build_kernel()
    N, C = 1, 1
    # Choose H and W so that HW*sizeof(float) is extremely high.
    # Note: Most GPUs have a limited shared memory per block (typically <= 48KB or 96KB).
    # Here we deliberately ask for a huge allocation and expect the kernel launch to fail.
    H, W = 1024, 1024  # 1024*1024*4 bytes ~ 4MB, which is far above shared memory limits.
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # This should raise a launch error due to excessive shared memory requirement.
        cuda_module.forward(x, weight, bias, 1e-5)
    torch.cuda.synchronize()

# Issue 4: The kernel does not safeguard against inputs with extremely large spatial dimensions.
# Although similar to issue 3, this test stresses that the kernel should ideally check input size.
def test_input_exceeding_device_shared_memory_limit():
    cuda_module = build_kernel()
    N, C = 1, 1
    # Choose H and W close to the maximum shared memory size.
    # For many devices the shared memory per block is < 100 KB.
    # Here, even 512*512 floats ~ 1MB is much larger.
    H, W = 512, 512  # 512*512*4 bytes ~ 1MB.
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Expect a failure when the shared memory request exceeds the device limits.
        cuda_module.forward(x, weight, bias, 1e-5)
    torch.cuda.synchronize()
