
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Utility: rebuild the extension.
def build_kernel(extra_cuda_cflags=None):
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags if extra_cuda_cflags is not None else ["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that passing non-float32 tensors (e.g. double) leads to unexpected behavior.
def test_input_tensor_type_issue():
    # Create double precision inputs.
    N = 1024
    pred = torch.randn(N, device="cuda", dtype=torch.double)
    targ = torch.randn(N, device="cuda", dtype=torch.double)
    mod = build_kernel()
    # Since the kernel blindly casts the data pointer to float*,
    # the resulting output will be wrong (very likely far off).
    output = mod.forward(pred, targ)
    torch.cuda.synchronize()
    # Compute reference with torch's smooth_l1_loss on double data.
    output_ref = torch.nn.functional.smooth_l1_loss(pred, targ)
    # Expect a huge difference.
    assert not torch.allclose(output, output_ref.float(), atol=1e-6), \
           "Kernel unexpectedly produced matching results for double precision input!"

# Issue 2: Test that a kernel launch using a block size exceeding the assumed shared memory size yields an error.
def test_block_size_issue():
    # Rebuild the module with an extra define to force a larger block size.
    # We trick the kernel launch by replacing the hardcoded block size.
    # In the kernel, block_size is hardcoded to 512.
    # We simulate a “mistaken” launch by compiling a modified version that uses a macro BLOCK_SIZE.
    # For this test, we require that kernel.cu be modified in our build for testing purposes:
    # it should use "const int block_size = BLOCK_SIZE;" rather than 512 if BLOCK_SIZE is defined.
    # (This test serves to trigger what would be unsafe if the kernel were launched with more than 512 threads.)
    extra_flags = ["-O3", "--use_fast_math", "-DBLOCK_SIZE=1024"]
    try:
        mod = build_kernel(extra_cuda_cflags=extra_flags)
    except Exception as e:
        # Compilation failed. That is consistent with the issue regarding using min and hardcoded shared memory.
        pytest.skip("Compilation with BLOCK_SIZE override failed; cannot test block size issue.")
    # Create an input where the number of elements is large enough.
    N = 1024 * 10
    pred = torch.randn(N, device="cuda", dtype=torch.float32)
    targ = torch.randn(N, device="cuda", dtype=torch.float32)
    # When launch is forced with 1024 threads per block but the shared memory is allocated for only 512 threads,
    # this should either cause a runtime error or lead to a result that is far from the expected value.
    output = mod.forward(pred, targ)
    torch.cuda.synchronize()
    output_ref = torch.nn.functional.smooth_l1_loss(pred, targ)
    # We check that the error is large since reduction over out‐of‐bounds shared memory is expected.
    assert torch.abs(output - output_ref) > 1e-2, "Kernel output is too close to reference despite block size issue."

# Issue 3: Test that the module compiles successfully. If the unqualified use of min caused a compile error,
# this test would not run. (Thus, success here indicates that at least in our environment the usage of min is accepted.)
def test_min_compilation_issue():
    try:
        mod = build_kernel()
    except Exception as e:
        pytest.fail(f"Failed to compile kernel.cu due to unqualified use of min: {e}")
    # A simple call to the forward function is enough.
    N = 1024
    pred = torch.randn(N, device="cuda", dtype=torch.float32)
    targ = torch.randn(N, device="cuda", dtype=torch.float32)
    output = mod.forward(pred, targ)
    torch.cuda.synchronize()
    ref = torch.nn.functional.smooth_l1_loss(pred, targ)
    assert torch.allclose(output, ref, atol=1e-5), "Kernel output mismatch in compilation test."

# Issue 4: Test for potential precision errors when many partial results are accumulated via atomicAdd.
def test_atomic_add_precision_issue():
    # Create a very large input to force many blocks and many atomicAdds.
    N = 1024 * 1024  # 1M elements
    pred = torch.randn(N, device="cuda", dtype=torch.float32)
    targ = torch.randn(N, device="cuda", dtype=torch.float32)
    mod = build_kernel()
    output = mod.forward(pred, targ)
    torch.cuda.synchronize()
    ref = torch.nn.functional.smooth_l1_loss(pred, targ)
    # Although small differences are always possible, in the presence of a reduction precision issue
    # the error may be significantly larger than expected.
    diff = torch.abs(output - ref)
    assert diff > 1e-5, f"Atomic add precision issue not observed: diff={diff.item()} (expected a larger difference due to reduction order effects)"
