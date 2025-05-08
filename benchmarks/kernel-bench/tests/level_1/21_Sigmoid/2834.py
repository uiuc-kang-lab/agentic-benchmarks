
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the CUDA module.
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compile_error():
    # Issue 1: The duplicated 'const' should prevent compilation.
    # We expect a RuntimeError (or similar compilation error) when the extension is built.
    with pytest.raises(RuntimeError):
        build_kernel()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_misaligned_tensor():
    # Issue 2: Create a tensor that is likely misaligned.
    # One way is to allocate a larger tensor and then slice it so that the underlying pointer
    # offset does not fall on a 16-byte boundary.
    kernel = build_kernel()
    base = torch.randn(17, 1024, device="cuda", dtype=torch.float32)
    # Slice off the first element along the last dimension to force misalignment.
    x = base[:, 1:]
    try:
        y = kernel.forward(x)
        # We cannot easily verify correctness when alignment is breached,
        # so if it runs without crashing, we flag a warning (or in a real test we'd check output).
    except Exception as e:
        pytest.fail(f"Kernel crashed with misaligned tensor: {e}")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unsupported_dtype():
    # Issue 3: The kernel’s AT_DISPATCH_FLOATING_TYPES macro covers only float and double,
    # so providing a half-precision tensor should trigger an error.
    kernel = build_kernel()
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        kernel.forward(x)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_tensor():
    # Issue 4: Create a noncontiguous tensor (e.g., by a transpose)
    kernel = build_kernel()
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    x_noncontig = x.transpose(0, 1)  # This makes the tensor noncontiguous.
    # Expect the kernel to either crash or throw an exception because it assumes contiguous memory.
    with pytest.raises(RuntimeError):
        kernel.forward(x_noncontig)
