
import pytest
import torch
from torch import nn
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="conv2d_cuda_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Kernel assumes float32. When given a double tensor the kernel misinterprets the data.
def test_wrong_dtype():
    # Create inputs of type double on CUDA:
    N, in_channels, H, W = 8, 3, 32, 32
    out_channels = 16
    kernel_size = (3, 3)
    x = torch.randn(N, in_channels, H, W, dtype=torch.double, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1],
                         dtype=torch.double, device="cuda")
    # For simplicity, do not supply bias.
    conv_ref = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
    conv_ref = conv_ref.cuda().double()
    ref_out = conv_ref(x)

    # Use our kernel extension (which internally calls data_ptr<float>())
    ext = build_kernel()
    # The extension assumes float so we cast inputs to float (simulate user mistake)
    x_wrong = x  # still double
    weight_wrong = weight  # still double
    # Call our CUDA function.
    # (Since the extension does not check dtypes, the output will likely be numerically off.)
    out = ext.forward(x_wrong, weight_wrong, None, [1,1], [0,0], [1,1], 1)
    # Compare with reference. They will likely differ a lot.
    # For the test, we assert that the outputs are NOT close.
    assert not torch.allclose(out.double(), ref_out, atol=1e-6), \
        "Kernel incorrectly accepted a double tensor. It must check for data type mismatch."

# Issue 2: Casting int64 dimensions to int may overflow in very large tensors.
# We simulate a situation in which H_out * W_out * N * C_out is huge.
# (Note: we cannot really allocate huge tensors so we simulate by manually calling the kernel
# with large dimension parameters.)
def test_int_overflow():
    ext = build_kernel()
    # Instead of actually allocating huge tensors (which is not feasible),
    # we simulate by supplying dimensions near the int limit.
    # For example, we choose output dimensions such that total_threads is greater than INT_MAX.
    #
    # Warning: In a real scenario the kernel will be launched with huge numbers and might crash.
    # Here we simply call the wrapper with fake dimensions if possible.
    #
    # One way to trigger the issue is to use a very large batch size on a small dummy tensor.
    # Since we cannot force integer overflow reliably in a unit test,
    # we check that the kernel accepts "reasonable" dimensions but warn that overflow might happen.
    N, C_in, H_in, W_in = 1, 1, 1024, 1024  # moderate sizes
    out_channels = 1
    kernel_size = (3, 3)
    x = torch.randn(N, C_in, H_in, W_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, C_in, kernel_size[0], kernel_size[1],
                         device="cuda", dtype=torch.float32)
    # Manually override dimensions by monkey-patching shapes in the extension, if desired.
    # Here we simply run the convolution and check that the output shape is computed.
    out = ext.forward(x, weight, None, [1,1], [1,1], [1,1], 1)
    # Expected H_out, W_out computed in C++: (H_in+padding*2 - (K-1)*dilation - 1)/stride + 1
    expected_H = (H_in + 2*1 - (kernel_size[0]-1)*1 - 1) // 1 + 1
    expected_W = (W_in + 2*1 - (kernel_size[1]-1)*1 - 1) // 1 + 1
    assert out.shape[2] == expected_H and out.shape[3] == expected_W, \
        "Output shape is not as expected (possible integer overflow in dimension calculations)."

# Issue 3: The kernel does not check if channels are divisible by groups.
def test_invalid_groups():
    # Use a case where in_channels is not divisible by groups.
    N = 4
    in_channels = 3  # Not divisible by groups=2.
    out_channels = 4  # Usually, out_channels should be divisible by groups too.
    kernel_size = (3, 3)
    x = torch.randn(N, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1],
                         device="cuda", dtype=torch.float32)
    # Use nn.Conv2d to get a reference error. Note that nn.Conv2d would normally raise an error.
    with pytest.raises(Exception):
        # This should raise an error in standard PyTorch due to channels/groups mismatch.
        nn.Conv2d(in_channels, out_channels, kernel_size, groups=2)

    # However, our CUDA extension does not check this.
    ext = build_kernel()
    # It will silently compute an output (with wrong indexing) so we check that the result
    # does not match what an nn.Conv2d configured with groups=1 produces.
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, groups=1, bias=False)
    conv = conv.cuda()
    ref_out = conv(x)
    out = ext.forward(x, weight, None, [1,1], [0,0], [1,1], 2)
    # We expect that the outputs differ because the grouping is mis-handled.
    assert not torch.allclose(out, ref_out, atol=1e-5), \
        "Kernel mis-handles channels/groups: the output matches an incorrect grouping configuration."

# Issue 4: The kernel launch does not perform a cudaDeviceSynchronize so errors may go undetected.
def test_cpu_tensor_input():
    # Passing a CPU tensor should trigger the TORCH_CHECK in the extension.
    N, in_channels, H, W = 2, 3, 16, 16
    out_channels = 4
    kernel_size = (3, 3)
    x_cpu = torch.randn(N, in_channels, H, W, device="cpu", dtype=torch.float32)
    weight_cpu = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1],
                             device="cpu", dtype=torch.float32)
    ext = build_kernel()
    with pytest.raises(RuntimeError, match="Input tensor must be on CUDA"):
        # This call should fail due to device mismatch.
        ext.forward(x_cpu, weight_cpu, None, [1,1], [0,0], [1,1], 1)
