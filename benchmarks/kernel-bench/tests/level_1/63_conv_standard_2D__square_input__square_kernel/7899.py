
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper: compile the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Using a double tensor (float64) should trigger an error (issue 1)
def test_dtype_error():
    cuda_module = build_kernel()
    # create double tensors on GPU
    N, Cin, H, W = 2, 3, 16, 16
    Cout, K = 4, 3
    x = torch.randn(N, Cin, H, W, dtype=torch.double, device='cuda')
    weight = torch.randn(Cout, Cin, K, K, dtype=torch.double, device='cuda')
    bias = torch.randn(Cout, dtype=torch.double, device='cuda')
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, weight, bias, 1, 0, 1, 1)
        
# Test 2: Passing groups != 1 should raise a TORCH_CHECK error (issue 2)
def test_groups_not_supported():
    cuda_module = build_kernel()
    N, Cin, H, W = 2, 3, 16, 16
    Cout, K = 4, 3
    x = torch.randn(N, Cin, H, W, device='cuda', dtype=torch.float32)
    weight = torch.randn(Cout, Cin, K, K, device='cuda', dtype=torch.float32)
    bias = torch.randn(Cout, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError, match="groups != 1 not supported"):
        cuda_module.forward(x, weight, bias, 1, 0, 1, 2)

# Test 3: Using a non-square kernel should lead to out‐of‐bound accesses (issue 3).
# We simulate this by making the weight tensor non-square.
def test_non_square_kernel():
    cuda_module = build_kernel()
    N, Cin, H, W = 2, 3, 16, 16
    Cout = 4
    # Create a non-square kernel: weight shape (Cout, Cin, K, K+1)
    K = 3
    weight = torch.randn(Cout, Cin, K, K+1, device='cuda', dtype=torch.float32)
    x = torch.randn(N, Cin, H, W, device='cuda', dtype=torch.float32)
    bias = torch.randn(Cout, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # The kernel reads K from weight.size(2) (i.e. 3) but then uses K for both dimensions.
        cuda_module.forward(x, weight, bias, 1, 0, 1, 1)

# Test 4: Passing a non‐contiguous input tensor should trigger the CHECK_CONTIGUOUS macro (issue 4)
def test_non_contiguous_input():
    cuda_module = build_kernel()
    N, Cin, H, W = 2, 3, 16, 16
    Cout, K = 4, 3
    x = torch.randn(N, Cin, H, W, device='cuda', dtype=torch.float32)
    # make x non-contiguous by transposing channels and then transposing back partially
    x_nc = x.transpose(1, 2)
    weight = torch.randn(Cout, Cin, K, K, device='cuda', dtype=torch.float32)
    bias = torch.randn(Cout, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        cuda_module.forward(x_nc, weight, bias, 1, 0, 1, 1)

# Test 5: When using many input channels, the kernel will use atomicAdd for accumulation
# and the output may be non-deterministic (issue 5).
def test_atomic_add_nondeterminism():
    cuda_module = build_kernel()
    torch.manual_seed(42)
    # Choose a channel count that forces partitions > 1.
    N, Cin, H, W = 1, 40, 32, 32  # With CHANNEL_TILE==16, partitions = ceil(40/16) == 3.
    Cout, K = 2, 3
    x = torch.randn(N, Cin, H, W, device='cuda', dtype=torch.float32)
    weight = torch.randn(Cout, Cin, K, K, device='cuda', dtype=torch.float32)
    bias = torch.randn(Cout, device='cuda', dtype=torch.float32)
    # Run the kernel repeatedly and compare outputs.
    out1 = cuda_module.forward(x, weight, bias, 1, 0, 1, 1)
    out2 = cuda_module.forward(x, weight, bias, 1, 0, 1, 1)
    # Due to non-deterministic atomic adds, allow a small difference.
    # (In practice these differences are typically very small; here we set a loose bound.)
    assert not torch.allclose(out1, out2, atol=1e-6), "Atomic adds may be deterministic but expected small numerical differences."

# Test 6: The kernel uses __ldg without verifying proper alignment.
# We simulate a scenario where the input tensor has an offset such that its storage pointer is misaligned.
# (This is tricky to simulate in PyTorch but we can create a non-zero offset view using as_strided.)
def test_misaligned_input():
    cuda_module = build_kernel()
    N, Cin, H, W = 2, 3, 16, 16
    Cout, K = 4, 3
    # Create a contiguous tensor
    base = torch.randn(N, Cin, H+1, W+1, device='cuda', dtype=torch.float32)
    # Create a view that starts at an offset, still reported as contiguous but pointer is offset.
    x = base.narrow(2, 1, H).narrow(3, 1, W)
    weight = torch.randn(Cout, Cin, K, K, device='cuda', dtype=torch.float32)
    bias = torch.randn(Cout, device='cuda', dtype=torch.float32)
    # It is possible that misaligned accesses cause performance warnings or errors on some hardware.
    # In our test, we simply run the kernel. (In some cases one might check for performance counters.)
    out = cuda_module.forward(x, weight, bias, 1, 0, 1, 1)
    # Reference conv2d for sanity (allowing for slight differences due to atomic adds)
    conv_ref = torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=0)
    assert torch.allclose(out, conv_ref, atol=1e-4), "Output does not match reference (possible mis-alignment issue)."
