
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu.
# Note: Ensure that kernel.cu is in the same directory as this test file.
def build_kernel():
    cuda_module = load(
        name="l1_norm_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: When D < 32 the shared memory allocation becomes zero but shared memory is later accessed.
def test_shared_memory_insufficient_allocation():
    my_module = build_kernel()
    # Create an input tensor with very few columns so that threads = min(threads, D) becomes <32.
    # For example, D = 1.
    N = 4
    D = 1
    x = torch.randn(N, D, device="cuda", dtype=torch.float32)
    # We expect that the kernel will access shared memory out-of-bound.
    with pytest.raises(RuntimeError):
        # Since the kernel error may be asynchronous, force synchronization.
        out = my_module.forward(x)
        torch.cuda.synchronize()

# Issue 2: The kernel does not check for the input tensor type (float32).
def test_input_tensor_wrong_dtype():
    my_module = build_kernel()
    # Create a double tensor, which is not supported by the kernel.
    N = 16
    D = 128
    x = torch.randn(N, D, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        out = my_module.forward(x)
        torch.cuda.synchronize()

# Issue 3: The kernel assumes proper alignment for reinterpret_cast to float4.
# We craft an input tensor that is not 16-byte aligned.
# One way to simulate misalignment is to create a larger tensor and then take a narrow slice starting at an offset.
def test_input_tensor_alignment():
    my_module = build_kernel()
    N = 8
    D = 128
    # Create an extra column so that slicing later may cause misalignment.
    x_full = torch.randn(N, D+1, device="cuda", dtype=torch.float32)
    # Slicing to ignore the first column can break the alignment.
    x = x_full[:, 1:]
    # The kernel does not check alignment; it will use reinterpret_cast on improperly aligned memory.
    # Although misaligned accesses on recent GPUs may not crash, they can produce incorrect results.
    # We check for numerical discrepancy with a reference L1 normalization.
    out = my_module.forward(x)
    torch.cuda.synchronize()
    # Compute correct L1 normalization on CPU.
    ref = x / torch.sum(torch.abs(x), dim=1, keepdim=True).clamp(min=1e-12)
    # Allow for some numerical error.
    assert not torch.allclose(out, ref, atol=1e-5), "Kernel unexpectedly produced correct results for misaligned input!"

# Issue 4: The kernel launches with thread count based solely on D.
# This is more a design limitation than a runtime error.
# We can check that for large D the configuration is chosen as intended,
# but then verify that the output is numerically close enough to expected behavior.
def test_thread_configuration_limitation():
    my_module = build_kernel()
    # Choose D such that the recommended threads would be set high.
    N = 16
    D = 2048  # D >= 1024 so threads become 1024.
    x = torch.randn(N, D, device="cuda", dtype=torch.float32)
    out = my_module.forward(x)
    torch.cuda.synchronize()
    ref = x / torch.sum(torch.abs(x), dim=1, keepdim=True).clamp(min=1e-12)
    # While this test does not trigger a crash, it highlights that the configuration is suboptimal.
    # We simply check that results are computed.
    assert torch.allclose(out, ref, atol=1e-5), "Kernel output differs from expected reference output!"

# Additionally, a test to ensure that passing a CPU tensor triggers the CUDA check.
def test_input_not_on_cuda():
    my_module = build_kernel()
    N = 16
    D = 128
    x = torch.randn(N, D, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        out = my_module.forward(x)
