
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA kernel module from kernel.cu.
def build_kernel():
    # Assumes kernel.cu is in the same directory as this test file.
    src_path = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="min_reduce_combined",
        sources=[src_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function: reference min reduction along a given dimension.
def reference_min(input_tensor, dim):
    # torch.min returns a tuple (min, indices), we only compare values.
    return torch.min(input_tensor, dim=dim)[0]

# Test 1: Nonstandard strides (Issue 1)
def test_nonstandard_strides():
    # Create a tensor and then permute its dimensions.
    # For example, create a tensor of shape [4, 5, 6] then permute to [5, 4, 6].
    # We choose reduction dimension=0 on the permuted tensor.
    x = torch.randn(4, 5, 6, device='cuda')
    y = x.permute(1, 0, 2)  # shape becomes [5, 4, 6] but noncontiguous.
    # Normally forward() in the extension calls contiguous() so that the nonstandard
    # strides are lost. To simulate the issue we reintroduce nonstandard strides by
    # computing the output “by hand” on a tensor that has non-canonical memory layout.
    # We create a new tensor using as_strided with the original (wrong) strides.
    # For this test, we force the kernel to see a tensor with nonstandard underlying strides.
    size = list(y.size())
    # This “fake” contiguous tensor will have the same storage as y but with custom strides.
    fake_strided = torch.as_strided(y, size=size, stride=(1, size[0], 1))  # intentionally wrong
    # Our extension does an internal contiguous() call; so we simulate a bypass by directly calling
    # the kernel function on fake_strided (assuming the user removed the contiguous() call in a custom build).
    module = build_kernel()
    # Call the CUDA kernel with reduction on dim=0 (as computed by the extension)
    # For the purpose of testing, we assume the kernel’s “forward” function.
    # Note: In a true scenario, if the tensor is noncontiguous the kernel indexing logic will lead to wrong results.
    out = module.forward(fake_strided, 0)
    # Compute reference result manually on fake_strided after making it contiguous according to its storage layout.
    # Here we mimic the kernel’s assumed layout: [outer, r, inner] where
    # outer = 1, r = size[0], inner = size[1]*size[2].
    # Our fake_strided has shape [5,4,6] but the memory layout is not standard.
    # So the torch.min on fake_strided (using the default contiguous view) will differ.
    ref = reference_min(fake_strided.contiguous(), 0)
    # Because the kernel used its flawed indexing, we expect a mismatch.
    with pytest.raises(AssertionError):
        assert torch.allclose(out, ref, atol=1e-5), "Nonstandard strides not handled correctly."

# Test 2: Empty reduction dimension (Issue 2)
def test_empty_reduction_dimension():
    # Create a tensor which has an empty reduction dimension.
    x = torch.randn(3, 0, 5, device="cuda")
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect an error (or undefined behavior) when the reduction dimension is empty.
        module.forward(x, 1)
    # Alternatively, if the kernel does not raise an error, the result is undefined.
    # In that case, one may assert that the result is not equal to the proper torch.min result.
    
# Test 3: Half precision (FP16) support (Issue 3)
def test_half_precision():
    # Create a half precision tensor.
    x = torch.randn(8, 16, device="cuda", dtype=torch.float16)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect that using float16 will trigger an error due to use of std::numeric_limits with FP16.
        module.forward(x, 0)

# Test 4: Kernel launch configuration assumptions (Issue 4)
def test_launch_configuration():
    # Here we test with a tensor whose outer*inner does not “fill” many blocks.
    # For example, a tensor with a small number of warps.
    # Even if the kernel launches correctly, a mis-assumption in the threads-per-block
    # arithmetic may yield wrong results.
    # We choose a tensor shape such that:
    #   outer = 1, r = 40, inner = 1; so total threads = 1*1*32 = 32 (one warp).
    x = torch.randn(40, device="cuda").reshape(40)  # shape [40]
    # For our kernel, we need a multi-dim tensor. We reshape it to [1, 40, 1].
    x = x.view(1, 40, 1)
    module = build_kernel()
    out = module.forward(x, 1)
    ref = reference_min(x.contiguous(), 1)
    # If the launch configuration were wrong the kernel result would differ from the expected min.
    assert torch.allclose(out, ref, atol=1e-5), "Kernel launch configuration error: output mismatch!"

