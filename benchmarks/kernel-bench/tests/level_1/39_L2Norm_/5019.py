
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Load the kernel extension from kernel.cu.
    # Note: In a real-world setup, the build parameters might be modified
    # to test different launch configurations. Here, we load the module as is.
    kernel_module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return kernel_module

@pytest.fixture(scope="module")
def kernel():
    # Build and return the CUDA kernel extension
    return build_kernel()

def test_non_cuda_input(kernel):
    # Issue 2: The kernel assumes the input is a CUDA tensor.
    # Providing a CPU tensor should fail.
    x = torch.randn(16, 128, device="cpu")
    with pytest.raises(RuntimeError, match="Input must be a CUDA tensor"):
        kernel.forward(x)

def test_non_contiguous_input(kernel):
    # Issue 2: The kernel requires the input to be contiguous.
    # For example, a transposed tensor is not contiguous.
    x = torch.randn(16, 128, device="cuda")
    x_noncontig = x.t()  # This makes a 128 x 16 tensor which is non-contiguous for the expected layout.
    with pytest.raises(RuntimeError, match="Input must be contiguous"):
        kernel.forward(x_noncontig)

def test_non_2d_input(kernel):
    # Issue 2: The kernel is written for 2D inputs only.
    x = torch.randn(16, 3, 4, device="cuda")
    with pytest.raises(RuntimeError, match="Input must be 2D"):
        kernel.forward(x)

def test_shared_memory_block_configuration(monkeypatch, kernel):
    # Issue 1: The kernel uses a fixed shared memory array of size 32.
    # Although the current forward function launches 256 threads per block (8 warps),
    # we simulate a situation that would use more threads per block if the kernel
    # were to be generalized. This test will not directly crash (unless the GPU
    # memory corruption occurs), but it attempts to mimic the scenario.
    #
    # For a more flexible design, the kernel's host code should allow altering
    # the block configuration (i.e., number of threads) so that one can intentionally
    # trigger a configuration in which the number of warps exceeds 32.
    #
    # Here, we simulate this by creating a large second dimension,
    # even though the current hard-coded block setting remains 256.
    #
    # NOTE: Since the kernel launch parameters are fixed in the code, this test serves as
    # a reminder that further modifications are needed to support custom block
    # configurations when generalizing the kernel.
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    # We run the kernel and compare with PyTorch’s built-in L2 normalization.
    out = kernel.forward(x)
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    out_ref = x / (norm + 1e-12)
    # If the shared memory fix is not applied (i.e. more than 32 warps used),
    # the results may be incorrect. We use a looser tolerance here for demonstration.
    assert torch.allclose(out, out_ref, atol=1e-3), "Output differs from expected L2 normalization. This may be caused by an incorrect shared memory configuration."

def test_valid_input(kernel):
    # A sanity check: for valid, small-sized 2D contiguous CUDA input,
    # the new kernel should produce the correct normalized output.
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32)
    out = kernel.forward(x)
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    out_ref = x / (norm + 1e-12)
    assert torch.allclose(out, out_ref, atol=1e-6), "L2 normalization output does not match expected result."
