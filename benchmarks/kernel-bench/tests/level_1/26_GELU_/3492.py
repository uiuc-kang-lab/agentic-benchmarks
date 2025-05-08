
import pytest
import torch
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

# Issue 1: Non-contiguous tensor input causes the kernel to iterate over memory incorrectly.
def test_non_contiguous_input():
    # Create a contiguous input and then make it non-contiguous by transposing
    x = torch.randn(64, 128, device="cuda")
    x_t = x.t()  # non-contiguous view
    my_module = build_kernel()
    
    # Run the CUDA kernel on the non-contiguous tensor.
    out = my_module.forward(x_t)
    # For comparison, compute reference GELU on a properly contiguized tensor and then view as the original shape.
    out_ref = torch.nn.functional.gelu(x_t.contiguous()).view_as(x_t)
    
    # We expect the results to be different because the kernel incorrectly assumes contiguity.
    if torch.allclose(out, torch.nn.functional.gelu(x_t)):
        pytest.fail("Kernel did not trigger the issue with non-contiguous inputs (it produced results as if the tensor were contiguous).")

# Issue 2: Kernel does not support half precision (float16)
def test_half_precision_input():
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect a runtime error because the kernel dispatch does not handle float16.
        my_module.forward(x)

# Issue 3: Grid dimension overrun.
def test_excessive_grid_dimension():
    # We choose a tensor size that forces the computed number of blocks to exceed typical CUDA limits.
    # Typical maximum grid dimension in x is 65535, so we craft a tensor such that:
    # blocks = ceil(numel / threads) > 65535 where threads = 256.
    threads = 256
    max_grid = 65535
    # set blocks to max_grid + 1
    blocks = max_grid + 1
    numel = threads * blocks
    # This tensor size is large but should be allocatable (e.g., for float32, numel ~ 16M elements)
    x = torch.randn(numel, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(Exception):
        # Expect an error because the grid dimensions exceed the maximum allowed.
        my_module.forward(x)

# Issue 4: Lack of explicit synchronization might delay detection of asynchronous errors.
# It is hard to trigger an asynchronous error from the kernel directly, but we can force a synchronous check.
def test_no_explicit_synchronization():
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    # Launch normally, then immediately force a device synchronization.
    out = my_module.forward(x)
    # Forcing synchronization to help catch asynchronous errors if any.
    torch.cuda.synchronize()
    # Compare with PyTorch's own GELU; while this test should pass if there are no errors,
    # it might reveal latent asynchronous problems.
    out_ref = torch.nn.functional.gelu(x)
    if not torch.allclose(out, out_ref, atol=1e-5):
        pytest.fail("Kernel output differs from expected output, indicating a possible asynchronous error.")

if __name__ == "__main__":
    pytest.main([__file__])
