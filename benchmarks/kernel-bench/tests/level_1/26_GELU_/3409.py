
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from the kernel.cu file.
def build_kernel():
    cuda_module = load(
        name="gelu_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference GELU implementation (using torch.nn.functional.gelu)
def ref_gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x)

# Test case 1: Misaligned memory pointer.
# We force misalignment by creating a tensor with extra elements and then taking a slice that does not start at an address
# that is 16-byte aligned.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_misalignment():
    # Create a tensor that is contiguous
    N = 1024 + 1  # +1 so that a subsequence may be misaligned
    x = torch.randn(N, dtype=torch.float32, device="cuda")
    # Slicing starting from index 1 likely yields a tensor whose data_ptr is not divisible by 16.
    # Note: This misalignment problem may not always crash but can produce wrong results.
    x_misaligned = x.narrow(0, 1, 1023)
    cuda_module = build_kernel()
    try:
        y_cuda = cuda_module.forward(x_misaligned)
    except RuntimeError as e:
        # If misaligned access triggers an error, then the issue is detected.
        pytest.skip(f"Kernel raised RuntimeError on misaligned input (as expected): {e}")
    # Compare to reference
    y_ref = ref_gelu(x_misaligned)
    # The results may differ substantially if misalignment corrupts memory reads.
    assert not torch.allclose(y_cuda, y_ref, atol=1e-4), (
        "Expected wrong result for misaligned input due to undefined behavior, but kernel output was close to reference."
    )

# Test case 2: Non-contiguous input tensor.
# We generate a non-contiguous tensor (e.g. via transpose) and invoke the kernel.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_non_contiguous():
    # Create a 2D tensor, then transpose to make it non-contiguous.
    x = torch.randn(128, 1024, dtype=torch.float32, device="cuda")
    x_non_contig = x.t()  # this tensor is non-contiguous
    # Although the actual number of elements is the same, the memory layout is not contiguous.
    cuda_module = build_kernel()
    try:
        y_cuda = cuda_module.forward(x_non_contig)
    except RuntimeError as e:
        pytest.skip(f"Kernel raised RuntimeError on non-contiguous input (as expected): {e}")
    y_ref = ref_gelu(x_non_contig)
    # Expect differences because reinterpret_cast based on a contiguous layout is invalid.
    assert not torch.allclose(y_cuda, y_ref, atol=1e-4), (
        "Kernel produced results similar to torch.gelu for a non-contiguous input, which is unexpected given the unsafe memory access."
    )

# Test case 3: Lack of proper error synchronization.
# We test that input validation in the kernel works by passing an input with the wrong type.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_wrong_dtype():
    # Pass a double tensor instead of float32.
    x = torch.randn(1024, dtype=torch.float64, device="cuda")
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError, match="Only float32 is supported"):
        cuda_module.forward(x)

# Test case 4: Passing a CPU tensor should trigger the CUDA check.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_cpu_tensor():
    x = torch.randn(1024, dtype=torch.float32, device="cpu")
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor"):
        cuda_module.forward(x)
