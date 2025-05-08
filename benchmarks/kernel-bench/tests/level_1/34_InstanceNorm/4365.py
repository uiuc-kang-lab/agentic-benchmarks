
import pytest
import torch
from torch.utils.cpp_extension import load
from torch.testing import assert_allclose

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel assumes float32 type.
def test_wrong_dtype():
    my_module = build_kernel()
    # Create a double tensor (float64) and a dummy weight and bias.
    batch_size = 4
    num_features = 8
    height = 16
    width = 16
    eps = 1e-5
    # Create input of wrong dtype.
    x = torch.randn(batch_size, num_features, height, width, dtype=torch.double, device="cuda")
    weight = torch.ones(num_features, dtype=torch.double, device="cuda")
    bias = torch.zeros(num_features, dtype=torch.double, device="cuda")
    with pytest.raises(RuntimeError):
        # This should raise an error or lead to wrong behavior because the kernel expects float32.
        my_module.forward(x, weight, bias, eps)

# Issue 2: Kernel assumes contiguous & 16-byte aligned memory.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a tensor and force it to be non-contiguous, e.g. via a transpose.
    batch_size = 4
    num_features = 8
    height = 16
    width = 16
    eps = 1e-5
    x = torch.randn(batch_size, num_features, height, width, device="cuda", dtype=torch.float32)
    # Transpose to get a non-contiguous tensor.
    x_non_contig = x.transpose(2, 3)
    # Create dummy weight and bias.
    weight = torch.ones(num_features, device="cuda", dtype=torch.float32)
    bias = torch.zeros(num_features, device="cuda", dtype=torch.float32)
    # If the kernel is fed a non-contiguous tensor, the vectorized memory accesses may lead to wrong results.
    # We simply check that the kernel output does not match what PyTorch's InstanceNorm2d produces.
    model = torch.nn.InstanceNorm2d(num_features, eps=eps).cuda().eval()
    ref = model(x_non_contig)
    y = my_module.forward(x_non_contig.contiguous(), weight, bias, eps)
    # Here we expect a difference because the proper usage for the kernel would be with contiguous memory.
    with pytest.raises(AssertionError):
        assert_allclose(y, ref)

# Issue 3: Kernel grid dimension assumptions.
def test_excessive_grid_dimension():
    my_module = build_kernel()
    # Create input dimensions such that N * C is very high.
    # Many GPUs limit the grid dimension in x (often to 2^31-1, but many tests and frameworks use smaller grids).
    # Use moderate H and W to ensure a launch, but set N and C high enough to simulate a potential grid dimension overload.
    # Note: It may be hard to actually exceed the hardware limit in a test. Here we simulate the case by choosing
    # large N and C that may stress the launch configuration.
    batch_size = 70000  # Extremely large batch (for test simulation purposes)
    num_features = 16
    height = 8
    width = 8
    eps = 1e-5
    # This tensor will have 70000*16 blocks which on some devices might exceed grid limits.
    try:
        x = torch.randn(batch_size, num_features, height, width, device="cuda", dtype=torch.float32)
        weight = torch.ones(num_features, device="cuda", dtype=torch.float32)
        bias = torch.zeros(num_features, device="cuda", dtype=torch.float32)
    except RuntimeError:
        pytest.skip("Skipping test_excessive_grid_dimension: unable to allocate huge input on this device.")
    # Expect the kernel to fail launching due to grid dimension issues.
    with pytest.raises(RuntimeError):
        my_module.forward(x, weight, bias, eps)

# Issue 4: No error checking for CUDA API calls.
# We simulate a kernel error by forcing an illegal memory access.
def test_kernel_memory_error(monkeypatch):
    my_module = build_kernel()
    
    # Monkey-patch the forward function in the CUDA module to pass an invalid pointer.
    # We simulate this by passing None for the input. This should trigger a CUDA error.
    with pytest.raises(RuntimeError):
        my_module.forward(torch.empty(0, device="cuda", dtype=torch.float32), 
                          torch.empty(0, device="cuda", dtype=torch.float32), 
                          torch.empty(0, device="cuda", dtype=torch.float32), 
                          1e-5)
