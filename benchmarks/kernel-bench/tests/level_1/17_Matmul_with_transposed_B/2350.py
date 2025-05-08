
import os
import re
import pytest
import torch
from torch.utils.cpp_extension import load

# Function to build and load our CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="matmul_transposed_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that passing tensors of type double does not produce correct results.
def test_input_tensor_dtype():
    # Build the module
    mod = build_kernel()
    M, K, N = 64, 37, 45  # arbitrary sizes (can be non-multiples of TILE_SIZE too)
    # Create double tensors on CUDA
    A = torch.randn(M, K, dtype=torch.double, device="cuda")
    B = torch.randn(N, K, dtype=torch.double, device="cuda")
    # The original kernel expects float and will use data_ptr<float>(),
    # so we force conversion to double—even though the kernel will misinterpret the bits.
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    # Compute the reference result with proper type conversions:
    C_ref = torch.matmul(A, B.T)
    # Due to type misinterpretation the result is expected to be wrong.
    # We assert that they are not close; if they were close, then the kernel would be
    # (accidentally) working for double inputs.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly produced correct results for double inputs!"

# Issue 2: Test that the kernel rejects batched (i.e. >2D) inputs.
def test_batched_input_error():
    mod = build_kernel()
    M, K, N = 32, 50, 40
    # Create a batched tensor (3D) for A and B
    A = torch.randn(2, M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(2, N, K, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        # The kernel code TORCH_CHECK(A.dim() == 2, ...) should trigger
        mod.forward(A, B)

# Issue 3: Test to check for redundant "#pragma unroll" directives in the source code.
def test_redundant_pragma_unroll():
    # Assume the kernel source code is in a file named "kernel.cu" in the current directory.
    kernel_path = os.path.join(os.path.dirname(__file__), "kernel.cu") if "__file__" in globals() else "kernel.cu"
    with open(kernel_path, "r") as f:
        source = f.read()
    # Count occurrences of "#pragma unroll" (ignoring case and whitespace)
    unroll_matches = re.findall(r"#pragma\s+unroll", source, flags=re.IGNORECASE)
    # We expect it to appear only once per loop. The redundant double unroll should be flagged.
    # Here we check if there is any instance where two unroll pragmas appear consecutively.
    if "pragma unroll" in source:
        # Look for two consecutive occurrences (possibly on consecutive lines)
        consecutive = re.search(r"(#pragma\s+unroll\s*\n\s*){2,}", source, flags=re.IGNORECASE)
        assert consecutive, "Did not detect redundant consecutive '#pragma unroll' directives, but they were expected."

# Issue 4: Test that the kernel produces the correct result when matrix sizes are not multiples of TILE_SIZE.
def test_non_multiple_dimensions():
    mod = build_kernel()
    # Choose sizes that are not multiples of TILE_SIZE (TILE_SIZE = 16)
    M, K, N = 30, 45, 17
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(N, K, dtype=torch.float32, device="cuda")
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    # Expected: A * B.T because in the Python code B is transposed
    C_ref = torch.matmul(A, B.T)
    assert torch.allclose(C, C_ref, atol=1e-5), f"Kernel output differs from reference output! Max difference: {(C-C_ref).abs().max().item()}"

# Issue 5: Test that the kernel checks for inner-dimension mismatches.
def test_mismatched_inner_dimensions():
    mod = build_kernel()
    # Create matrices with mismatched inner dimensions.
    # A is of shape (M, K) and B is of shape (N, K+1) so that A.size(1) != B.size(1)
    M, K, N = 32, 50, 40
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(N, K + 1, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        mod.forward(A, B)
