
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Helper function to compile and load the CUDA extension module.
def build_kernel():
    # Force recompilation for testing.
    module = load(
        name="gelu_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# ============================
# Test 1: Input tensor type check
# This test attempts to pass a non-float32 tensor through the kernel. Since the kernel code
# only supports float32, the output will be unpredictable or wrong.
def test_input_tensor_type():
    my_module = build_kernel()
    # Create a double precision tensor on CUDA
    x = torch.randn(2000, 2000, dtype=torch.double, device="cuda")
    # Expect that using a double tensor will lead to a runtime error
    with pytest.raises(RuntimeError):
        # Even if it doesn't throw immediately, the results will be numerically incorrect.
        # Depending on timing the kernel might crash. We use a context manager to catch an error.
        y = my_module.forward(x)
        torch.cuda.synchronize()

# ============================
# Test 2: Misaligned memory due to slicing
# This test creates a tensor and then uses a slice so that the underlying data pointer is offset.
# That offsets the pointer so that it is not 16-byte aligned, which should trigger the vectorized
# load/store issue. We then compare the custom kernel result with PyTorch's GELU.
def test_misaligned_memory():
    my_module = build_kernel()
    N = 1005
    # Create a larger tensor so that after slicing, the data pointer is misaligned.
    big = torch.randn(N + 10, dtype=torch.float32, device="cuda")
    # Create a view that is offset by one element. The new data pointer will be original pointer plus 4 bytes.
    x = big.narrow(0, 1, N)
    # Verify misalignment: pointer mod 16 should not be 0.
    if x.data_ptr() % 16 == 0:
        pytest.skip("Tensor is unexpectedly aligned; cannot test misalignment")
    # Compute GELU using the custom kernel.
    y_kernel = my_module.forward(x)
    # Compute GELU using PyTorch (using the same formulation as in the original Python model)
    sqrt_2_over_pi = (2.0 / torch.pi).sqrt()
    coeff = 0.044715
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    y_ref = 0.5 * x * (1.0 + torch.tanh(inner))
    torch.cuda.synchronize()
    # The results may differ significantly if misaligned accesses produce undefined behavior.
    # We expect a noticeable difference.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), \
        "Kernel output unexpectedly matches reference output despite misaligned memory"

# ============================
# Test 3: Kernel launch configuration / error checking
# This test uses a very large tensor that forces the kernel launch configuration to hit the max blocks cap.
# Because the launch configuration hard-caps the grid at 1024 blocks, the kernel may silently produce
# incorrect results or run suboptimally.
def test_kernel_grid_limit():
    my_module = build_kernel()
    # Create a very large tensor.
    N = 1024 * 256 + 13  # a number large enough to exceed 1024 blocks when divided by block size 256
    x = torch.randn(N, dtype=torch.float32, device="cuda")
    y_kernel = my_module.forward(x)
    # Compute GELU using PyTorch's formulation.
    sqrt_2_over_pi = (2.0 / torch.pi).sqrt()
    coeff = 0.044715
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    y_ref = 0.5 * x * (1.0 + torch.tanh(inner))
    torch.cuda.synchronize()
    # This test does not insist on exact matching, rather it detects if the kernel output is far off.
    # If the grid configuration is insufficient then some elements might not be processed correctly.
    diff = (y_kernel - y_ref).abs().max().item()
    assert diff > 1e-3, \
        "Kernel output is too close to reference; the grid limit issue did not manifest as expected."

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
