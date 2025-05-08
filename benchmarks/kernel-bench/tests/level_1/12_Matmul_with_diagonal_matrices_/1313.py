
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Utility function to build the kernel extension.
def build_kernel(extra_cuda_cflags=None, source_replacements=None):
    # Read the original source code from kernel.cu
    with open("kernel.cu", "r") as f:
        code = f.read()
    
    # If modifications in the source are needed, perform string replacements.
    # This can be used to simulate non-standard block dimensions.
    if source_replacements is not None:
        for old, new in source_replacements.items():
            code = code.replace(old, new)
        # Write modified code to a temporary file.
        src_file = "kernel_modified.cu"
        with open(src_file, "w") as f:
            f.write(code)
    else:
        src_file = "kernel.cu"
    
    # Build the module.
    cuda_module = load(
        name="diag_matmul_module",
        sources=[src_file],
        extra_cuda_cflags=extra_cuda_cflags if extra_cuda_cflags else ["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    
    return cuda_module

# Test for issue 1:
# The test will force a block configuration that is non-conforming with respect
# to the warp broadcast assumption. In the original kernel the block dimension along x
# is hardcoded to 32. We override it to 16 (with a text substitution) so that a row’s threads
# will be split among more than one warp. Then we compare the kernel output with the reference.
def test_nonstandard_block_dim():
    # We modify the kernel source to force block_dim_x = 16 instead of 32.
    replacements = {
        "const int block_dim_x = 32;": "const int block_dim_x = 16;"
    }
    module = build_kernel(source_replacements=replacements)
    
    # Create small input matrices so that the grid will include several blocks.
    N = 128
    M = 128
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, M, device="cuda", dtype=torch.float32)
    
    # Compute reference result
    # Using the diagonal multiplication: diag(A) @ B is equivalent to multiplying each row of B by A[i]
    C_ref = A.unsqueeze(1) * B
    
    # Call our modified kernel
    C = module.forward(A, B)
    torch.cuda.synchronize()
    
    # Because of the warp-broadcast bug this test is expected to *fail* (the output will differ).
    # We assert that the maximum absolute difference is greater than some threshold.
    max_diff = (C - C_ref).abs().max().item()
    assert max_diff > 1e-3, f"Expected error due to warp broadcast issues, but max diff was {max_diff}"

# Test for issue 2:
# The kernel only supports float32. We test that providing double precision input raises an error.
def test_input_tensor_dtype():
    module = build_kernel()
    
    N = 128
    M = 128
    A = torch.randn(N, device="cuda", dtype=torch.float64)  # double precision
    B = torch.randn(N, M, device="cuda", dtype=torch.float64)  # double precision
    
    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel expects float inputs.
        C = module.forward(A, B)
        torch.cuda.synchronize()
