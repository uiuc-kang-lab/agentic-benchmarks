
import pytest
import torch
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="efficient_avg_pool2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_grid():
    # Issue 1: Grid-dimension limitation
    # Query the device properties. The maximum allowed grid dim in z can be low (often 65535).
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    max_grid_z = props.maxGridSize[2]  # typically 65535

    # Choose N and C so that N * C > max_grid_z.
    # We keep H and W small to avoid huge memory allocations.
    N = max_grid_z // 10 + 1   # force product to be > max_grid_z when multiplied by C
    C = 10
    H = 16
    W = 16
    kernel_size = 3
    stride = kernel_size
    padding = 0

    # Create a tensor on CUDA.
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    module = build_kernel()
    # Expect the kernel launch to fail due to grid z-dim limitation.
    with pytest.raises(RuntimeError) as excinfo:
        output = module.forward(x, kernel_size, stride, padding)
        torch.cuda.synchronize()
    assert "gridDim" in str(excinfo.value) or "launch" in str(excinfo.value), \
        "Kernel did not error with large grid dimension as expected."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_half_precision():
    # Issue 2: Lack of half-precision support.
    device = torch.device("cuda")
    N, C, H, W = 8, 4, 32, 32
    kernel_size = 3
    stride = kernel_size
    padding = 0

    # Create a half-precision tensor.
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float16)
    module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        # The dispatch does not include half-precision, so this should throw an error.
        output = module.forward(x, kernel_size, stride, padding)
        torch.cuda.synchronize()
    assert "dispatch" in str(excinfo.value) or "not implemented" in str(excinfo.value), \
        "Kernel did not complain about half precision as expected."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_runtime_kernel_size_unrolling():
    # Issue 3: Loop unrolling when kernel_size is not 3.
    # Use a common kernel size that is not 3 (e.g., 4) and test that the result matches nn.AvgPool2d.
    device = torch.device("cuda")
    N, C, H, W = 4, 3, 32, 32
    kernel_size = 4
    stride = 2
    padding = 1

    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    module = build_kernel()
    output_cuda = module.forward(x, kernel_size, stride, padding)
    torch.cuda.synchronize()

    # Use PyTorch's built in AvgPool2d for reference.
    avgpool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    output_ref = avgpool(x)
    assert torch.allclose(output_cuda, output_ref, atol=1e-5), \
        "Output from the generic loop 'unroll' path does not match the reference AvgPool2d result."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    # Issue 4: Assumption of contiguous memory layout.
    # Create an input tensor that is not contiguous and bypass the contiguous() call by simulating a scenario
    # where the caller accidentally passes a non-contiguous tensor.
    # For this test, we create a non-contiguous view and then manually call the kernel (simulating a bad usage).
    device = torch.device("cuda")
    N, C, H, W = 4, 3, 32, 32
    kernel_size = 3
    stride = kernel_size
    padding = 0

    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    # Create a non-contiguous tensor by transposing dimensions.
    x_noncontig = x.transpose(2,3)
    assert not x_noncontig.is_contiguous(), "The tensor should be non-contiguous for this test."

    module = build_kernel()
    # Even though the forward() of the module calls contiguous(), here we simulate an API that passes non-contiguous tensor.
    # We directly call the kernel function assuming the contiguous() call is missing.
    # To simulate this, we pass the non-contiguous tensor's pointer (which is a mistake).
    # Since our kernel assumes standard indexing, the result is likely to be incorrect.
    output_cuda = module.forward(x_noncontig, kernel_size, stride, padding)
    torch.cuda.synchronize()

    # Reference using contiguous version.
    avgpool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    output_ref = avgpool(x_noncontig.contiguous())
    # We expect a difference due to non-contiguous layout access in the kernel.
    assert not torch.allclose(output_cuda, output_ref, atol=1e-4), \
        "Kernel output should be wrong when a non-contiguous tensor is passed without proper handling."
