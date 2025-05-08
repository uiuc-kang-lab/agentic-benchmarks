
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Test 1: Pass a non-2D tensor (e.g. a 3D tensor) to trigger the 2D-tensor check.
def test_non_2d_input():
    cuda_module = build_kernel()
    # Create a 3D tensor which should cause a TORCH_CHECK failure since the kernel expects 2D.
    x = torch.randn(4, 5, 6, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Expected 2D tensor for this example."):
        cuda_module.forward(x)
        
# Test 2: Pass a tensor with a data type other than float32 to trigger misinterpretation.
def test_input_tensor_dtype():
    cuda_module = build_kernel()
    # Create a tensor with double precision. The kernel will treat it as float32,
    # so the results will be wrong.
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float64)
    # Compute a reference normalization in double precision.
    ref = x / torch.sum(torch.abs(x), dim=1, keepdim=True)
    out = cuda_module.forward(x)
    torch.cuda.synchronize()
    # The results are expected to be quite off due to the type mismatch
    assert not torch.allclose(out, ref, atol=1e-5), "Kernel unexpectedly produced correct results with double dtype."

# Test 3: Pass a tensor with D == 0 (zero columns) to trigger the potential kernel launch issue.
def test_zero_dimension():
    cuda_module = build_kernel()
    batch_size = 16
    dim = 0  # Zero feature dimension
    x = torch.empty(batch_size, dim, device="cuda", dtype=torch.float32)
    # Expect an error (or misbehavior) because the kernel does not handle D == 0.
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(x)
        torch.cuda.synchronize()

# Test 4: Check for silent kernel launch issues by triggering invalid launch configuration indirectly.
# For example, if D is very small, the computed thread count becomes D (< warp size), and while that may work,
# it might hide issues if kernel error checking were added. In our case we try to force a scenario where the kernel
# might misbehave due to lack of error checking.
def test_small_dimension():
    cuda_module = build_kernel()
    batch_size = 16
    dim = 5  # small dimension, where many threads in the block will do no work.
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32)
    # Compute expected normalized result.
    ref = x / torch.sum(torch.abs(x), dim=1, keepdim=True)
    out = cuda_module.forward(x)
    torch.cuda.synchronize()
    # While the kernel might operate correctly, the lack of error checking makes detecting problems harder.
    # Here we at least check that the result is not numerically too far off.
    assert torch.allclose(out, ref, atol=1e-5), f"Output is not as expected for a small dimension. Max diff: {(out - ref).abs().max()}"
