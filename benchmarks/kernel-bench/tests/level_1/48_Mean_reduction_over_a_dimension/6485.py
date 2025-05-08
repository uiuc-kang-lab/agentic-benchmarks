
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="mean_reduce_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Non-contiguous tensor input leads to incorrect indexing.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    # Create a tensor and make it non-contiguous by transposing two dimensions.
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.float32)
    x_t = x.transpose(1, 2)  # This makes the tensor non-contiguous.
    # Use reduction over dimension 1 (which is the original dim2)
    # Get reference result from torch.mean (which handles non-contiguous tensors)
    ref = torch.mean(x_t, dim=1)
    # Call our CUDA extension kernel (which assumes a contiguous layout)
    # Expect the output to differ from the reference result.
    out = cuda_module.forward(x_t, 1)
    torch.cuda.synchronize()
    # The error threshold is set very small so that any difference is caught.
    assert not torch.allclose(out, ref, atol=1e-5), \
        f"Kernel unexpectedly produced the correct result on non-contiguous input!"

# Issue 2: Reduced dimension of size zero leads to a division-by-zero error.
def test_zero_dim_reduction():
    cuda_module = build_kernel()
    # Create a tensor where the reduction dimension has size 0.
    x = torch.empty(16, 0, 256, device="cuda", dtype=torch.float32)
    with pytest.raises(Exception):
        # Depending on CUDA behavior, this might raise an error (e.g. division by zero).
        _ = cuda_module.forward(x, 1)
        torch.cuda.synchronize()

# Issue 3: Input tensor of an unsupported type (e.g. integer) is not handled.
def test_unsupported_input_type():
    cuda_module = build_kernel()
    # Create an integer tensor.
    x = torch.randint(0, 10, (16, 256, 256), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel only dispatches floating point types.
        _ = cuda_module.forward(x, 1)
        torch.cuda.synchronize()
