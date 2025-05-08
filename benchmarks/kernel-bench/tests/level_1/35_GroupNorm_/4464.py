
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the extension module from the C++/CUDA file "kernel.cu"
    cuda_module = load(
        name="group_norm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_divisible_channels():
    # Issue 1: When channels are NOT divisible by num_groups the kernel’s integer division
    # (channels_per_group = C // num_groups) will lead to a mis‐mapping.
    #
    # For example, set C = 10 and num_groups = 3.
    # PyTorch's nn.GroupNorm will by design throw an exception, but our kernel does not check
    # this (and may compute a result, but that result is simply wrong).
    batch_size = 2
    channels = 10
    height, width = 4, 4
    num_groups = 3  # 10 % 3 != 0, so GroupNorm cannot be validly defined
    eps = 1e-5

    device = "cuda"
    # Create input and parameter tensors
    x = torch.randn(batch_size, channels, height, width, device=device)
    weight = torch.randn(channels, device=device)
    bias = torch.randn(channels, device=device)

    # Build the module
    mod = build_kernel()

    # Call the kernel. Since the kernel does not check the divisibility condition,
    # it will run and produce some output even though the grouping is ill‐defined.
    out = mod.forward(x, weight, bias, num_groups, eps)

    # For comparison, try to create a torch.nn.GroupNorm with these parameters.
    # This should raise an error. If it does not, then the kernel’s behavior is not consistent.
    with pytest.raises(Exception):
        torch.nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=eps)

    # Also, since the kernel computed something, we can check that at least one channel is wrong.
    # For example, we can compute normalization for the first 9 channels by manually emulating
    # the “truncated” grouping and show that channel 9 (the extra channel) is not normalized correctly.
    # (This test is admittedly heuristic.)
    normalized_part = (x[:, :9] - x[:, :9].mean(dim=[1,2,3], keepdim=True)) / torch.sqrt(x[:, :9].var(dim=[1,2,3], keepdim=True) + eps)
    # The output from the kernel for these channels would have the affine parameters applied.
    # We check that the output in the extra channel is different from what an ideal GN would produce.
    diff = (out[:,9].abs()).mean().item()
    assert diff > 1e-3, f"Output of ill‐defined channel normalization seems too close to zero; diff={diff}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_contiguous_input():
    # Issue 2: The kernel assumes contiguous memory. If a non-contiguous tensor is passed,
    # the flattened indexing used in the kernel will produce incorrect results.
    batch_size = 4
    channels = 32
    height, width = 16, 16
    num_groups = 8
    eps = 1e-5

    device = "cuda"
    # Generate a contiguous input tensor and corresponding parameters.
    x_contig = torch.randn(batch_size, channels, height, width, device=device)
    weight = torch.randn(channels, device=device)
    bias = torch.randn(channels, device=device)

    # Build the extension module.
    mod = build_kernel()

    # First, run the kernel with a contiguous input
    out_contig = mod.forward(x_contig, weight, bias, num_groups, eps)

    # Now create a non-contiguous view by permuting dimensions (note that the expected shape is (N, C, *))
    # so we need to bring it back to shape (N, C, *) but memory layout is altered.
    x_noncontig = x_contig.permute(0, 2, 3, 1)  # shape becomes (N, H, W, C) and is non-contiguous
    # Reshape back into (N, C, H, W) without calling contiguous()
    x_noncontig = x_noncontig.view(batch_size, channels, height, width)

    out_noncontig = mod.forward(x_noncontig, weight, bias, num_groups, eps)

    # For a correct kernel that ignores layout issues, the results should match.
    # Here, however, because x_noncontig is non-contiguous, the results are likely to deviate.
    if torch.allclose(out_contig, out_noncontig, atol=1e-5):
        pytest.fail("Kernel output is (unexpectedly) identical for non-contiguous and contiguous inputs")

