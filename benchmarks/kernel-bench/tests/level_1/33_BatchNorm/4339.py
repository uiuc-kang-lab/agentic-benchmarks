
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to compile the CUDA kernel from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# A helper BatchNorm function using the custom CUDA kernel interface.
def run_batch_norm(kernel_module, input, weight, bias, running_mean, running_var, training, momentum, eps):
    return kernel_module.forward(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )

# Issue 1:
# Test that passing a non-float32 (double) tensor leads to wrong results.
def test_non_float32_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    kernel_module = build_kernel()
    
    # Use double precision input (even though the kernel expects float)
    batch_size, C, H, W = 16, 16, 32, 32
    x = torch.randn(batch_size, C, H, W, dtype=torch.double, device='cuda')
    weight = torch.randn(C, dtype=torch.double, device='cuda')
    bias = torch.randn(C, dtype=torch.double, device='cuda')
    running_mean = torch.zeros(C, dtype=torch.double, device='cuda')
    running_var = torch.ones(C, dtype=torch.double, device='cuda')
    training = True
    momentum = 0.1
    eps = 1e-5

    # The kernel will reinterpret the double memory as float memory.
    out = run_batch_norm(kernel_module, x, weight, bias, running_mean, running_var, training, momentum, eps)
    # Compute reference using PyTorch's BatchNorm2d (casting x to float for reference)
    x_float = x.float()
    bn = torch.nn.BatchNorm2d(C, momentum=momentum, eps=eps).to('cuda')
    # Set weight, bias, running stats
    bn.weight.data.copy_(weight.float())
    bn.bias.data.copy_(bias.float())
    bn.running_mean.copy_(running_mean.float())
    bn.running_var.copy_(running_var.float())
    ref_out = bn(x_float)
    # The outputs should differ significantly due to wrong interpretation of data.
    assert not torch.allclose(out, ref_out, atol=1e-3), "Kernel incorrectly handled non-float32 input."

# Issue 2:
# Test that passing a tensor with an unsupported number of dimensions (e.g. 3D tensor)
# triggers an indexing error.
def test_non_4d_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    kernel_module = build_kernel()
    
    # Create a 3D tensor (simulate BatchNorm1d input) instead of 4D.
    batch_size, C, L = 16, 16, 64
    x = torch.randn(batch_size, C, L, device='cuda', dtype=torch.float32)
    weight = torch.randn(C, device='cuda', dtype=torch.float32)
    bias = torch.randn(C, device='cuda', dtype=torch.float32)
    running_mean = torch.zeros(C, device='cuda', dtype=torch.float32)
    running_var = torch.ones(C, device='cuda', dtype=torch.float32)
    training = True
    momentum = 0.1
    eps = 1e-5

    with pytest.raises(IndexError):
        # The kernel expects four dimensions and will try to access input.size(2) as H and input.size(3) as W.
        run_batch_norm(kernel_module, x, weight, bias, running_mean, running_var, training, momentum, eps)

# Issue 3:
# Test that using a block configuration with fewer than 2 warps causes an out-of-bounds shared memory access.
# In this test, we force the kernel to use a reduced thread count.
# We compile a temporary kernel with a modified thread configuration.
def test_inadequate_block_size():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Manually compile a variant of the kernel with a smaller block size by passing an extra flag.
    # Here, we simulate a scenario by modifying the kernel launch parameters via a dummy kernel.
    # For simplicity, we assume that the user might recompile with threads=32, which gives 1 warp.
    cuda_module = load(
        name="test_module_small_block",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_include_paths=[],
        with_cuda=True,
        verbose=False,
    )

    # We'll simulate the effect by calling the kernel launcher with a 3D input,
    # but then override the kernel launch parameters on the Python side.
    batch_size, C, H, W = 16, 8, 32, 32  # use small C to simplify
    x = torch.randn(batch_size, C, H, W, device='cuda', dtype=torch.float32)
    weight = torch.randn(C, device='cuda', dtype=torch.float32)
    bias = torch.randn(C, device='cuda', dtype=torch.float32)
    running_mean = torch.zeros(C, device='cuda', dtype=torch.float32)
    running_var = torch.ones(C, device='cuda', dtype=torch.float32)
    training = True
    momentum = 0.1
    eps = 1e-5

    # We mimic an inadequate block size by overriding the kernel launch parameters.
    # WARNING: This test is expected to cause an illegal memory access or wrong behavior.
    try:
        # Launch kernel with only 32 threads (1 warp). The kernel assumes at least 2 warps when writing to
        # shared memory indices warp_sums[0] and warp_sums[1].
        threads = 32
        num_warps = threads // 32  # equals 1, which is insufficient
        shared_mem_bytes = 2 * num_warps * torch.finfo(torch.float32).bits // 8  # compute bytes
        # Directly call the kernel with a modified grid: one block per channel.
        # Use the internal kernel function from the loaded module.
        cuda_module.batch_norm_modular_kernel[
            (C,), (threads,), shared_mem_bytes
        ](
            x.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr(),
            running_mean.data_ptr(),
            running_var.data_ptr(),
            training,
            momentum,
            eps,
            torch.empty_like(x).data_ptr(),
            x.size(0),
            x.size(1),
            x.size(2),
            x.size(3)
        )
        # If the kernel does not crash, we check if the running_mean is not updated correctly.
        torch.cuda.synchronize()
        # Inappropriate shared memory access with 1 warp is expected to yield wrong output.
        # Here we simply fail the test because the kernel should not work with insufficient warps.
        pytest.fail("Kernel did not crash or produce an error with an inadequate block size configuration.")
    except RuntimeError as e:
        # We expect a runtime error due to illegal memory access.
        assert "illegal memory access" in str(e) or "out of bounds" in str(e), f"Unexpected error message: {str(e)}"
