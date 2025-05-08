
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension containing the kernel (from kernel.cu)
def build_kernel():
    cuda_module = load(
        name="instance_norm_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper: call our forward function from the cuda module.
def run_instance_norm(x, weight, bias, eps=1e-5):
    module = build_kernel()
    # The forward function from the module is expected to call our kernel.
    y = module.forward(x, weight, bias, float(eps))
    # We synchronize in the Python test to catch any errors from kernel launch.
    torch.cuda.synchronize()
    return y

# Issue 1: Non-16-byte aligned tensor.
# We simulate misalignment by creating a non-contiguous view.
def test_non_aligned_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Create a contiguous input tensor and then get a non-contiguous slice.
    x_full = torch.randn(4, 8, 16, 16, device="cuda", dtype=torch.float32)
    # Slicing along the channel dimension makes the tensor non-contiguous.
    x_non_contig = x_full[:, 1: , :, :]
    # Use standard weight and bias.
    C = x_non_contig.size(1)
    weight = torch.ones(C, device="cuda", dtype=torch.float32)
    bias = torch.zeros(C, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match=".*misaligned.*|.*invalid.*"):
        # Expecting a runtime error or CUDA error because of misalignment.
        run_instance_norm(x_non_contig, weight, bias)

# Issue 2: Block reduction assumption on block size.
# Although BLOCK_SIZE is fixed in the kernel compile, we trigger the potential issue
# by calling the kernel on an input with a very large spatial size that forces many iterations.
def test_large_block_usage():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Create an input where H*W is huge and not multiples of vector size in some regions.
    # (e.g., 1x1 with spatial dimensions not divisible by 4).
    x = torch.randn(2, 16, 103, 103, device="cuda", dtype=torch.float32)
    weight = torch.ones(16, device="cuda", dtype=torch.float32)
    bias = torch.zeros(16, device="cuda", dtype=torch.float32)
    # In a more general scenario, a user might choose a larger block size. Here,
    # the test is to see that the reduction does not cause shared memory overruns.
    # Since the kernel does not dynamically check block sizes, we simply check for CUDA errors.
    try:
        run_instance_norm(x, weight, bias)
    except RuntimeError as e:
        pytest.fail(f"Kernel reduction issue triggered an error: {e}")

# Issue 3: Non-contiguous input memory.
def test_non_contiguous_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Transpose the input to force non-contiguous memory
    x = torch.randn(2, 16, 32, 32, device="cuda", dtype=torch.float32).transpose(1, 2)
    # Even though the kernel expects a contiguous layout, users in more general scenarios
    # may pass non-contiguous inputs. We expect either an error or incorrect results.
    weight = torch.ones(16, device="cuda", dtype=torch.float32)
    bias = torch.zeros(16, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match=".*non-contiguous.*|.*unexpected layout.*"):
        run_instance_norm(x, weight, bias)

# Issue 4: Lack of kernel launch error checking.
# We test a situation that should normally work but then we deliberately pass an invalid epsilon
# to see if the kernel launch error is caught on host side.
def test_invalid_epsilon():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.randn(2, 16, 32, 32, device="cuda", dtype=torch.float32)
    weight = torch.ones(16, device="cuda", dtype=torch.float32)
    bias = torch.zeros(16, device="cuda", dtype=torch.float32)
    # Use a negative epsilon, which may lead to NaNs in rsqrt.
    with pytest.raises(RuntimeError, match=".*numerical error.*|.*invalid.*"):
        run_instance_norm(x, weight, bias, eps=-1e-5)
