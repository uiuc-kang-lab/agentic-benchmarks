
import os
import shutil
import tempfile
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to build the original module (which assumes 256 threads per block).
def build_kernel():
    cuda_module = load(
        name="gelu_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# For issue 1, we simulate a scenario where the block size is not a multiple of 32.
# We do this by modifying the kernel source to use a macro for threads per block.
def build_kernel_non_multiple_warp(threads_per_block):
    # Create a temporary directory to hold the modified kernel file.
    temp_dir = tempfile.mkdtemp()
    src_path = os.path.join(temp_dir, "kernel.cu")
    # Read the original kernel.cu file.
    with open("kernel.cu", "r") as f:
        source = f.read()
    # Replace the fixed threads_per_block value with a macro value.
    # The source contains: "const int threads_per_block = 256;"
    # We replace it with: "const int threads_per_block = THREADS_PER_BLOCK;"
    modified_source = source.replace("const int threads_per_block = 256;", 
                                     "const int threads_per_block = THREADS_PER_BLOCK;")
    with open(src_path, "w") as f:
        f.write(modified_source)
    cuda_module = load(
        name="gelu_module_custom",
        sources=[src_path],
        extra_cuda_cflags=["-O3", "--use_fast_math", f"-DTHREADS_PER_BLOCK={threads_per_block}"],
        with_cuda=True,
        verbose=True,
        build_directory=temp_dir,
    )
    # Clean up the temporary directory after loading.
    shutil.rmtree(temp_dir)
    return cuda_module

# Issue 2: Non‑contiguous tensor input.
def test_non_contiguous_input():
    module = build_kernel()
    # Create a contiguous tensor and then make it non‑contiguous by transposing.
    # For example, create a 2D tensor then take its transpose.
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # Transpose makes it non-contiguous (if not square)
    # Run our CUDA kernel.
    y = module.forward(x_noncontig)
    # Compare to PyTorch's gelu (which supports non-contiguous inputs correctly).
    y_ref = torch.nn.functional.gelu(x_noncontig)
    # The two outputs should be close if the kernel had properly handled strides.
    # We expect a mismatch because the kernel incorrectly assumes contiguity.
    assert not torch.allclose(y, y_ref, atol=1e-5), (
        "Test failed: when given a non-contiguous input tensor, the kernel produced output matching "
        "PyTorch's gelu. The kernel should not work correctly on non-contiguous inputs."
    )

# Issue 3: Passing a non‑floating point tensor.
def test_non_floating_point_type():
    module = build_kernel()
    # Create an integer tensor on CUDA.
    x = torch.randint(0, 10, (128,), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="gelu_cuda"):
        # Expect the AT_DISPATCH macro to throw an error because int32 is not a supported type.
        module.forward(x)

# Issue 1: Block size not a multiple of 32.
# We build a version of the kernel that uses a custom block size (e.g. 250).
def test_blocksize_non_multiple_of_warp():
    # Use 250 threads per block which is not divisible by 32.
    module_custom = build_kernel_non_multiple_warp(250)
    # Create an input tensor with an arbitrary number of elements.
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    y = module_custom.forward(x)
    y_ref = torch.nn.functional.gelu(x)
    # Due to the bug, some elements are never computed correctly. Thus, the output will differ from the reference.
    assert not torch.allclose(y, y_ref, atol=1e-5), (
        "Test failed: using a block size not divisible by 32, the kernel should produce incorrect results "
        "but it matched the reference output."
    )
