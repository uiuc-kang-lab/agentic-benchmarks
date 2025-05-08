
import os
import shutil
import tempfile
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper: load the normal kernel extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test for Issue 1: Hard-coded reduction array size (assuming blockDim.x <= 256)
# This test creates a temporary copy of kernel.cu in which the block size is forced to 512,
# but the reduction array is still declared as __shared__ float s_sum[256];
# Using 512 threads per block will cause out-of-bound accesses and yield a wrong result.
def test_large_block_dim():
    # Create a temporary directory and make a copy, modifying the host function's block size.
    temp_dir = tempfile.mkdtemp()
    try:
        src_path = os.path.join(os.getcwd(), "kernel.cu")
        temp_kernel = os.path.join(temp_dir, "kernel_custom.cu")
        with open(src_path, "r") as fin, open(temp_kernel, "w") as fout:
            for line in fin:
                # Replace the constant block size in the host launcher.
                # (forced to 512 instead of 256)
                if "const int block_size = 256;" in line:
                    line = "    const int block_size = 512;\n"
                # Also, change the static shared memory array used in the reduction.
                if "__shared__ float s_sum[256];" in line:
                    line = "    __shared__ float s_sum[512]; // changed size but kernel reduction loop remains unchanged\n"
                fout.write(line)
        # Build the extension from the modified file.
        my_module = load(
            name="test_module_large", 
            sources=[temp_kernel],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            with_cuda=True,
            verbose=True,
        )
        # Create input where total number of elements equals the forced block size.
        # In this configuration, one block should run and use 512 threads.
        N = 512
        predictions = torch.randn(N, device="cuda", dtype=torch.float32)
        targets = torch.randn(N, device="cuda", dtype=torch.float32)
        # Compute the expected loss using PyTorch’s reference implementation.
        expected = torch.nn.functional.smooth_l1_loss(predictions, targets)
        # Get the kernel result.
        out = my_module.forward(predictions, targets)
        torch.cuda.synchronize()
        # Because the reduction code now incorrectly assumes 256 threads, the loss will be computed incorrectly.
        # In particular, the difference in the output will be larger than a small tolerance.
        diff = (out - expected).abs().item()
        assert diff > 1e-3, f"Test for large block dim did not trigger the issue: diff={diff}"
    finally:
        shutil.rmtree(temp_dir)

# Test for Issue 2: Division-by-zero when n_elements==0.
def test_empty_input():
    # Create empty (n_elements==0) tensors for predictions and targets.
    predictions = torch.tensor([], device="cuda", dtype=torch.float32)
    targets = torch.tensor([], device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    # The kernel will do atomicAdd(output, s_sum[0] / 0) causing an illegal division.
    # In many cases the floating-point division by zero in GPU code results in NaN.
    # We therefore expect that the output contains NaN.
    out = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    # Check that the result is NaN.
    assert torch.isnan(out).all(), "Expected NaN output when input tensor is empty (division by zero)."

# Test for Issue 3: Kernel not supporting types other than float32.
def test_non_float_input():
    my_module = build_kernel()
    # Create double precision tensors.
    predictions = torch.randn(128, 4096, dtype=torch.double, device="cuda")
    targets = torch.randn(128, 4096, dtype=torch.double, device="cuda")
    with pytest.raises(RuntimeError) as excinfo:
        # The host wrapper checks for device type and contiguity,
        # but not for the floating-point type. Since the kernel is written only for float,
        # this should raise an error.
        my_module.forward(predictions, targets)
    assert "Expected" in str(excinfo.value) or "float" in str(excinfo.value), (
        "Expected a type compatibility error when input tensors are not float32."
    )

# (Optionally) Test for error when inputs are non-contiguous.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create contiguous tensors and then make them non-contiguous.
    a = torch.randn(130, 4096, device="cuda", dtype=torch.float32)
    b = torch.randn(130, 4096, device="cuda", dtype=torch.float32)
    predictions = a.transpose(0, 1)
    targets = b.transpose(0, 1)
    # The host wrapper requires contiguous tensors.
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(predictions, targets)
    assert "contiguous" in str(excinfo.value), (
        "Expected an error about non-contiguous tensors."
    )
