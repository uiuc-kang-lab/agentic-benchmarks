
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="gn_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Reference GroupNorm using PyTorch
class RefGroupNorm(nn.Module):
    def __init__(self, num_features, num_groups, eps):
        super().__init__()
        self.gn = nn.GroupNorm(num_channels=num_features, num_groups=num_groups, eps=eps)
    def forward(self, x):
        return self.gn(x)

# Test 1:
# Trigger issue (1): finalize_stats_kernel_atomic missing bounds check.
# Use an input whose total number of groups is small,
# so that when we compute "blocks = (total_groups + threads - 1) // threads",
# many extra threads (with idx >= total_groups) will be launched.
@pytest.mark.cuda
def test_finalization_bounds():
    # Use a very small input so that N*num_groups is much smaller than the block size (256)
    # For example: N=1 and C=num_groups=3, so total groups = 3 but 256 threads are launched.
    N, C, H, W = 1, 3, 1, 1
    num_groups = 3
    eps = 1e-5
    device = torch.device("cuda")
    
    # Create input and scale parameters
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    weight = torch.randn(C, device=device, dtype=torch.float32)
    bias = torch.randn(C, device=device, dtype=torch.float32)

    # Build our custom CUDA forward function
    gn_cuda = build_kernel()
    y_cuda = gn_cuda.forward(x, weight, bias, num_groups, eps)
    
    # Compute reference using PyTorch's GroupNorm
    ref_gn = RefGroupNorm(num_features=C, num_groups=num_groups, eps=eps).to(device).eval()
    with torch.no_grad():
        y_ref = ref_gn(x)
    
    # The missing bounds check in the finalize kernel is likely to lead to incorrect results
    # (or NaNs) compared to the reference.
    # We assert that the outputs are NOT all close, indicating the bug is triggered.
    assert not torch.allclose(y_cuda, y_ref, atol=1e-5), \
        "Test did not trigger the expected bounds error in finalize_stats_kernel_atomic."

# Test 2:
# Trigger issue (4): non-divisible channel count groups.
# Here C is not a multiple of num_groups,
# so that channels_per_group = C // num_groups ignores leftover channels.
@pytest.mark.cuda
def test_non_divisible_channels():
    # Here 10 channels with 3 groups: 10 // 3 = 3 channels per group (with one channel left out).
    N, C, H, W = 1, 10, 8, 8
    num_groups = 3
    eps = 1e-5
    device = torch.device("cuda")
    
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    weight = torch.randn(C, device=device, dtype=torch.float32)
    bias = torch.randn(C, device=device, dtype=torch.float32)
    
    gn_cuda = build_kernel()
    y_cuda = gn_cuda.forward(x, weight, bias, num_groups, eps)
    
    ref_gn = RefGroupNorm(num_features=C, num_groups=num_groups, eps=eps).to(device).eval()
    with torch.no_grad():
        try:
            y_ref = ref_gn(x)
        except Exception as e:
            # PyTorch's GroupNorm will raise an error if channels are not divisible by groups.
            y_ref = None
    
    # If our kernel does not check the remainder channels,
    # then its output will not match the reference behavior (error or different normalization).
    # We require that they are NOT close.
    if y_ref is not None:
        assert not torch.allclose(y_cuda, y_ref, atol=1e-5), \
            "Test did not trigger the expected behavior for non-divisible channel counts."
    else:
        # If the reference raises an error, then the extension should not be accepted as correct.
        pytest.fail("Reference GroupNorm raised an error, indicating non-divisible channels are not handled.")

# Test 3:
# Trigger potential issues (2,3,5) by trying to run the extension with double (float64) data.
# On some architectures double atomicAdd is not supported. In that case, the custom kernel may error.
@pytest.mark.cuda
def test_double_dtype():
    N, C, H, W = 1, 4, 8, 8
    num_groups = 2
    eps = 1e-5
    device = torch.device("cuda")
    
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float64)
    weight = torch.randn(C, device=device, dtype=torch.float64)
    bias = torch.randn(C, device=device, dtype=torch.float64)
    
    gn_cuda = build_kernel()
    with pytest.raises(Exception):
        # Expect that running the kernel with double (which uses atomicAdd on doubles)
        # may raise an error on devices without full double precision atomic support.
        _ = gn_cuda.forward(x, weight, bias, num_groups, eps)
