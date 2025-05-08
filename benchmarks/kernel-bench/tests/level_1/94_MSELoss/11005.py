
import pytest
import torch
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_double_input_issue():
    # Issue 1: The kernel improperly vectorizes loads using float2 even for double data type.
    # Prepare double type inputs with an even number of elements.
    N = 1024
    # Create contiguous double tensors on CUDA.
    predictions = torch.randn(N, device="cuda", dtype=torch.double)
    targets = torch.randn(N, device="cuda", dtype=torch.double)

    module = build_kernel()
    # Execute kernel. Expected behavior is to compute mean squared error.
    # Since the kernel uses float2 vectorized loads unconditionally, the output is likely wrong.
    result = module.forward(predictions, targets)
    # Compute reference result using PyTorch (on CPU then send to cuda with same dtype for comparison)
    ref = torch.mean((predictions - targets) ** 2)
    # The result will be incorrect due to misinterpretation of double data.
    assert not torch.allclose(result, ref, atol=1e-5), (
        f"Kernel unexpectedly produced a correct result for double input."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_odd_element_issue():
    # Issue 2: The kernel assumes the number of elements is a multiple of 2.
    # Create input tensors with an odd number of elements.
    N = 1023  # odd number
    predictions = torch.randn(N, device="cuda", dtype=torch.float32)
    targets = torch.randn(N, device="cuda", dtype=torch.float32)

    module = build_kernel()
    # The kernel may perform out-of-bounds access due to vectorized loads.
    # We expect a runtime error or wrong result.
    with pytest.raises(RuntimeError):
        result = module.forward(predictions, targets)
        # Explicitly synchronize to force error detection
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_atomic_add_double_capability_issue():
    # Issue 3: atomicAdd for double is not supported on GPUs with compute capability < 6.0.
    # Check device capability.
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    if cap_major >= 6:
        pytest.skip("Device supports atomicAdd on double. Test only valid for older architectures.")
    
    # Create input tensors of type float32 (to avoid Issue 1) with even number of elements.
    N = 1024
    predictions = torch.randn(N, device="cuda", dtype=torch.float32)
    targets = torch.randn(N, device="cuda", dtype=torch.float32)

    module = build_kernel()
    # We construct a scenario where the accumulator tensor is double.
    # If the device does not support atomicAdd on double, a runtime error is expected.
    with pytest.raises(RuntimeError):
        result = module.forward(predictions, targets)
        torch.cuda.synchronize()
