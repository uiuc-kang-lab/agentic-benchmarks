
import torch
import pytest
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

# Issue 1:
# The kernel works only for float32 but does not check for other dtypes.
def test_wrong_dtype():
    module = build_kernel()
    batch_size, m, k, n = 2, 16, 16, 16
    # create double tensors (not supported by our kernel)
    A = torch.randn(batch_size, m, k, dtype=torch.double, device="cuda")
    B = torch.randn(batch_size, k, n, dtype=torch.double, device="cuda")
    # Although the CPU side math (torch.bmm) supports double, our kernel forces float*
    # The kernel may silently compute wrong values or crash.
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.bmm(A, B)
    # Expect the outputs to differ significantly.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel should not correctly process non-float32 input."

# Issue 2:
# The kernel assumes 128-bit aligned memory. We create misaligned tensors
# by slicing an otherwise contiguous tensor to force an unaligned pointer.
def test_non_aligned_memory():
    module = build_kernel()
    batch_size, m, k, n = 2, 33, 33, 33
    # Create a slightly larger tensor and then slice to force a misalignment
    # (by not starting at the original pointer)
    A_big = torch.randn(batch_size, m, k + 1, device="cuda", dtype=torch.float32)
    B_big = torch.randn(batch_size, k + 1, n, device="cuda", dtype=torch.float32)
    # Slicing off the first column in the inner dimension should offset the data pointer.
    A = A_big.narrow(2, 1, k).contiguous()
    B = B_big.narrow(1, 1, k).contiguous()
    
    # Run the custom kernel
    C = module.forward(A, B)
    torch.cuda.synchronize()
    # Run the reference torch.bmm
    C_ref = torch.bmm(A, B)
    # In case misalignment causes incorrect behavior, the outputs will diverge.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel expected to fail or produce different results with misaligned memory."

# Issue 3:
# The kernel does not check for errors after launch.
# To expose this, we force a situation where the kernel launch should fail.
# One way is to pass a non-contiguous tensor (which might lead to wrong memory accesses)
# even though our index computations expect contiguous layout.
def test_non_contiguous_tensor():
    module = build_kernel()
    batch_size, m, k, n = 2, 32, 32, 32
    A = torch.randn(batch_size, m, k, device="cuda", dtype=torch.float32)
    B = torch.randn(batch_size, k, n, device="cuda", dtype=torch.float32)
    # Make A non-contiguous by transposing the last two dimensions and then transposing back
    A_nc = A.transpose(1,2)
    A_nc = A_nc.transpose(1,2)  # still non-contiguous if the transposition does not revert to original memory layout
    assert not A_nc.is_contiguous(), "A_nc should be non-contiguous to trigger potential access errors."
    
    # Launch the kernel with a non-contiguous tensor. Although torch.bmm can sometimes handle it,
    # our kernel directly uses data_ptr() and assumes contiguous memory.
    with pytest.raises(AssertionError) as exc_info:
        # Compute both C and reference; the kernel may simply produce incorrect results.
        C = module.forward(A_nc, B)
        torch.cuda.synchronize()
        C_ref = torch.bmm(A_nc, B)
        # We force a check that detects the deviation.
        assert torch.allclose(C, C_ref, atol=1e-5), \
            f"Kernel output differs from reference output! Difference: {(C - C_ref).abs().max().item()}"
    
    # If no exception is raised, then silently fail the test.
    if exc_info.value is None:
        pytest.fail("Kernel did not detect issues with non-contiguous input as expected.")
