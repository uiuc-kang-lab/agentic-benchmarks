
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build and load the CUDA extension from kernel.cu.
def build_kernel():
    # Ensure that the absolute path is used in case the build system changes directories.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cuda_file = os.path.join(dir_path, "kernel.cu")
    module = load(
        name="test_module",
        sources=[cuda_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return module

# Issue 1: Test with misaligned memory.
# This creates a tensor whose storage is contiguous but we create a view that is offset,
# leading to a pointer that is not properly aligned for vectorized loads/stores.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_misaligned_memory():
    module = build_kernel()
    # Create a contiguous tensor then slice to misalign it.
    base = torch.randn(1000 + 1, dtype=torch.float32, device="cuda")
    # Slicing out the first element to possibly misalign the memory pointer.
    misaligned = base[1:]
    # The new tensor might not be aligned to 16 bytes required for float4.
    # Expect either a crash or wrong computation. Here we test if the result is not equal to torch.relu.
    with pytest.raises(RuntimeError):
        # If misalignment causes an access error, we expect a runtime error.
        output = module.forward(misaligned)
        torch.cuda.synchronize()

# Issue 2: Test with non-contiguous tensor.
# The kernel uses reinterpret_cast assuming contiguous memory, so non-contiguous inputs could break it.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_tensor():
    module = build_kernel()
    # Create a 2D tensor and then transpose it to make it non-contiguous.
    x = torch.randn(32, 32, dtype=torch.float32, device="cuda")
    non_contig = x.t()  # now non-contiguous.
    # Expect the kernel operation to fail or produce an incorrect result.
    with pytest.raises(RuntimeError):
        output = module.forward(non_contig)
        torch.cuda.synchronize()

# Issue 3: Test with an unsupported tensor type (half precision).
# The dispatch macro only supports float and double, so using half should trigger an error.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unsupported_dtype_half():
    module = build_kernel()
    x = torch.randn(1024, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        # Expect an error because half precision is not in AT_DISPATCH_FLOATING_TYPES.
        output = module.forward(x)
        torch.cuda.synchronize()

# Issue 4: Test with a CPU tensor.
# The kernel should be used on CUDA tensors. Passing a CPU tensor may lead to an error.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_tensor():
    module = build_kernel()
    x = torch.randn(1024, dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError):
        output = module.forward(x)
