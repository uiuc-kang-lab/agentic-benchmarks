
import math
import pytest
import torch
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

# Helper: reference Frobenius norm based normalization
def frobenius_normalize(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x, p='fro')
    # Protect against division by zero for the reference computation.
    return x if norm == 0 else x / norm

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_wrong_dtype():
    # Issue 1: The kernel assumes float32.
    # Create a tensor with an unsupported type (float64)
    x = torch.randn(16, 64, 32, 32, dtype=torch.float64, device="cuda")
    module = build_kernel()
    # Call forward; since the underlying kernel misinterprets double as float,
    # the output will differ from the reference.
    with pytest.raises(Exception) as excinfo:
        # the kernel does not explicitly check dtype, so it might produce garbage.
        # We treat a wrong result as a failure.
        out = module.forward(x)  
        torch.cuda.synchronize()
    # If no exception is raised, then compare output with reference
    # (most likely the output will not be correct)
    # The test fails if the output is (accidentally) close to the expected value.
    # (This test is designed to trigger the dtype issue.)
    # Note: Uncomment the assertion below if the kernel silently produces wrong output.
    # expected = frobenius_normalize(x.float())
    # assert not torch.allclose(out, expected, atol=1e-5), "Kernel incorrectly handled non-float32 input"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_empty_tensor():
    # Issue 2: Handling of empty tensors.
    x = torch.empty((0,), dtype=torch.float32, device="cuda")
    module = build_kernel()
    # When input is empty, norm will be computed on 0 elements.
    # Depending on how reduction is handled, this might trigger an error or produce NaN.
    out = module.forward(x)
    torch.cuda.synchronize()
    # The expected behavior is ambiguous; here we check that the output is empty.
    assert out.numel() == 0, f"Output tensor is expected to be empty but got shape: {out.shape}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_zero_tensor():
    # Issue 3: Division by zero if input tensor is all zeros.
    x = torch.zeros(16, 64, 32, 32, dtype=torch.float32, device="cuda")
    module = build_kernel()
    out = module.forward(x)
    torch.cuda.synchronize()
    # When norm is 0, division by zero will occur.
    # We expect the output to contain NaN or inf values.
    assert (torch.isnan(out) | torch.isinf(out)).all(), "Kernel did not produce NaN or Inf when normalizing zero tensor"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_tensor():
    # Issue 4 (partial): The kernel checks for contiguity and demands a contiguous input.
    x = torch.randn(16, 64, 32, 32, dtype=torch.float32, device="cuda")
    x_noncontig = x.transpose(1, 2)  # make tensor non-contiguous
    module = build_kernel()
    with pytest.raises(RuntimeError, match="Input must be contiguous"):
        # The forward function in the wrapper makes a TORCH_CHECK on contiguity.
        module.forward(x_noncontig)
    # This test ensures that non-contiguous tensors are properly rejected.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multistream_constant_memory_race():
    # Issue 5: Using __constant__ memory for d_norm without proper synchronization in concurrent contexts.
    # We simulate multiple concurrent calls by launching them in separate streams.
    module = build_kernel()
    streams = [torch.cuda.Stream() for _ in range(4)]
    results = []
    x = torch.randn(16, 64, 32, 32, dtype=torch.float32, device="cuda")
    expected = frobenius_normalize(x)
    # Launch the same module.forward in different streams.
    for s in streams:
        with torch.cuda.stream(s):
            res = module.forward(x)
            results.append(res)
    # Wait for all streams to finish.
    torch.cuda.synchronize()
    # Check that all outputs match the expected normalized tensor.
    for i, out in enumerate(results):
        if not torch.allclose(out, expected, atol=1e-4):
            pytest.fail(f"Stream {i}: Kernel output does not match expected normalized tensor, "
                        "indicating a potential race condition on constant memory usage.")
