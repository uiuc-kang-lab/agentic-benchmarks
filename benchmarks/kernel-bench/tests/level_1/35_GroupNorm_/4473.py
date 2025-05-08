
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Wrong behavior when the input tensor is not float32 (data type assumption)
def test_dtype_issue():
    my_module = build_kernel()
    batch_size, features, dim1, dim2 = 4, 16, 32, 32
    # Use double precision input (which is not supported by the vectorized float4 loads)
    x = torch.randn(batch_size, features, dim1, dim2, dtype=torch.float64, device='cuda')
    weight = torch.randn(features, device='cuda', dtype=torch.float64)
    bias = torch.randn(features, device='cuda', dtype=torch.float64)
    try:
        # This call may silently produce an incorrect result instead of an error.
        y = my_module.forward(x, weight, bias, num_groups=4, eps=1e-5)
    except Exception as e:
        pytest.skip("Kernel could not process double precision input: " + str(e))
    # Compute reference using PyTorch GroupNorm
    gn = torch.nn.GroupNorm(num_groups=4, num_channels=features, eps=1e-5).cuda().double()
    y_ref = gn(x)
    # Expect the results to be off due to the incorrect vectorized load interpretation.
    assert not torch.allclose(y, y_ref, atol=1e-3), "Kernel unexpectedly handled double precision correctly!"

# Issue 2: The kernel assumes aligned memory (misalignment issue)
def test_alignment_issue():
    my_module = build_kernel()
    batch_size, features, dim1, dim2 = 4, 16, 32, 32
    # Create a contiguous tensor first ...
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda')
    # ... then create a non-contiguous view by transposing two spatial dimensions.
    x_non_contig = x.transpose(2, 3)  # This view will be non–contiguous and likely mis–aligned.
    weight = torch.randn(features, device='cuda')
    bias = torch.randn(features, device='cuda')
    try:
        y = my_module.forward(x_non_contig, weight, bias, num_groups=4, eps=1e-5)
    except Exception as e:
        pytest.skip("Kernel did not handle non–contiguous input: " + str(e))
    # Calculate a reference using PyTorch’s native GroupNorm (which calls .contiguous() when necessary)
    gn = torch.nn.GroupNorm(num_groups=4, num_channels=features, eps=1e-5).cuda()
    y_ref = gn(x_non_contig)
    # Because the kernel assumes contiguous, its result is expected to be different.
    assert not torch.allclose(y, y_ref, atol=1e-3), "Kernel unexpectedly handled mis–aligned / non–contiguous input correctly!"

# Issue 3: The shared memory reduction uses a fixed array size of 32,
# so if more than 32 warps are present in a block (i.e. blockDim.x > 1024),
# the shared memory writes will go out-of-bounds.
@pytest.mark.xfail(reason="Cannot easily simulate blockDim > 1024 from Python; this test serves as a placeholder to highlight the potential issue.")
def test_shared_memory_issue():
    my_module = build_kernel()
    # Here we try to force a very large input, so that if someone attempts to launch the kernel
    # with a non–standard configuration (or mistakenly increases blockDim.x), the shared memory fixed size
    # becomes an issue. Note: the host code always uses 256 threads for compute_stats_kernel,
    # so we simulate this by providing a huge spatial size.
    batch_size, features, dim1, dim2 = 4, 2048, 64, 64
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda')
    weight = torch.randn(features, device='cuda')
    bias = torch.randn(features, device='cuda')
    y = my_module.forward(x, weight, bias, num_groups=8, eps=1e-5)
    gn = torch.nn.GroupNorm(num_groups=8, num_channels=features, eps=1e-5).cuda()
    y_ref = gn(x)
    # We expect differences because if a launch had >1024 threads per block the shared memory reduction might misbehave.
    assert not torch.allclose(y, y_ref, atol=1e-3), "Kernel unexpectedly handled excessive threads per block correctly!"

# Issue 4 (Control test): When a proper contiguous float32 tensor is provided, the kernel should work correctly.
def test_contiguous_input():
    my_module = build_kernel()
    batch_size, features, dim1, dim2 = 4, 16, 32, 32
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda')
    weight = torch.randn(features, device='cuda')
    bias = torch.randn(features, device='cuda')
    y = my_module.forward(x, weight, bias, num_groups=4, eps=1e-5)
    # Compute reference GroupNorm using PyTorch
    gn = torch.nn.GroupNorm(num_groups=4, num_channels=features, eps=1e-5).cuda()
    y_ref = gn(x)
    # With well–behaved contiguous input, the kernel should give results close to the reference.
    assert torch.allclose(y, y_ref, atol=1e-2), "Kernel output differs from PyTorch GroupNorm for contiguous input!"
