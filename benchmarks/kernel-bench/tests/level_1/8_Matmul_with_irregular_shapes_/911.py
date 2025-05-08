
import os
import re
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to build and load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel assumes 2D inputs.
# This test passes batched (3D) tensors to trigger the incompatibility.
def test_batched_input():
    # Create a batched input instead of 2D matrices.
    batch = 2
    M, K, N = 64, 32, 48
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, K, N, device="cuda", dtype=torch.float32)
    
    module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # Since the kernel expects A to be 2D, this should trigger an error.
        # We simulate the batched use-case by iterating over the batch:
        outputs = []
        for a, b in zip(A, B):
            outputs.append(module.forward(a, b))
        # If no error raised, check one output against torch.matmul (should fail if batched)
        C_ref = torch.matmul(A[0], B[0])
        torch.cuda.synchronize()
        assert torch.allclose(outputs[0], C_ref, atol=1e-5)

# Issue 2: Kernel does not validate input data type.
# This test passes double precision (float64) inputs to trigger an error.
def test_wrong_dtype():
    M, K, N = 128, 64, 96
    A = torch.randn(M, K, device="cuda", dtype=torch.float64)
    B = torch.randn(K, N, device="cuda", dtype=torch.float64)
    
    module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # The internal casting in matmul_cuda uses data_ptr<float>().
        # This mismatch in type is expected to lead to a runtime error.
        module.forward(A, B)

# Issue 3: __shared__ arrays declared inside loops.
# While this may not always crash at runtime, we can inspect the source code
# of the kernel to ensure that shared memory is not declared inside loops.
def test_shared_declaration_location():
    # Read the contents of kernel.cu and search for __shared__
    # declarations occurring within a loop.
    kernel_file = "kernel.cu"
    assert os.path.exists(kernel_file), "kernel.cu file not found!"
    with open(kernel_file, "r") as f:
        source = f.read()
    
    # A crude check: search for '__shared__' declarations that occur after a for( pattern.
    # This is not perfect but should catch declarations inside loops.
    pattern = r"for\s*\(.*\)\s*{[^}]*__shared__"
    found = re.search(pattern, source, re.DOTALL)
    assert found is None, "Shared memory arrays are declared inside a loop, which is incorrect. They should be declared at block scope."
