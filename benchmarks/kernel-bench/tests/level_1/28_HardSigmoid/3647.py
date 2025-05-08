
import torch
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

# Issue 1: Lack of support for half-precision.
def test_half_precision_unsupported():
    my_module = build_kernel()
    # Create a half precision (float16) tensor on CUDA.
    x = torch.randn(1024, device="cuda", dtype=torch.half)
    with pytest.raises(RuntimeError):
        # Expect the kernel to raise an error because half is not selected in the dispatch macro.
        my_module.forward(x)

# Issue 3: No proper device synchronization to catch errors.
def test_non_cuda_input():
    my_module = build_kernel()
    # Passing a CPU tensor should trigger the TORCH_CHECK that requires a CUDA tensor.
    x = torch.randn(1024, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        my_module.forward(x)

# Issue 2: Using int for indexing may overflow for very large tensors.
# NOTE: This test is conceptual because allocating a tensor with >INT_MAX elements is impractical.
# We simulate the effect by monkey-patching the tensor's numel() method.
@pytest.mark.skip(reason="Simulated overflow test: this test is conceptually triggering the bug and is skipped to avoid instability.")
def test_large_tensor_overflow():
    my_module = build_kernel()
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    # Save the original numel
    original_numel = x.numel
    try:
        # Monkey-patch numel() to simulate a huge number of elements (overflow scenario)
        x.numel = lambda: (2**31) + 10
        with pytest.raises(RuntimeError):
            my_module.forward(x)
    finally:
        # Restore the original numel method
        x.numel = original_numel

# Issue 4: Unnecessary use of shared memory for constant scalars.
# While this is mainly a performance/maintainability issue, we check that the kernel output is correct.
def test_hardsigmoid_correctness():
    my_module = build_kernel()
    x = torch.randn(2048, device="cuda", dtype=torch.float32)
    output = my_module.forward(x)
    expected = torch.nn.functional.hardsigmoid(x)
    assert torch.allclose(output, expected, atol=1e-5), "Kernel output does not match expected HardSigmoid output."
