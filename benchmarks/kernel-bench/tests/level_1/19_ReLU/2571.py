
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Compile the CUDA extension from kernel.cu
@pytest.fixture(scope="module")
def cuda_module():
    cuda_module = load(
        name="kernel_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_misaligned_memory(cuda_module):
    # Create a tensor of float32 that is padded by one element so that a sub-tensor is misaligned:
    # Allocate an extra element and then slice to force an offset.
    base = torch.randn(100 * 4 + 1, dtype=torch.float32, device="cuda")
    # Create a sub-tensor that is not 16-byte aligned.
    misaligned = base.narrow(0, 1, 100 * 4)
    # Even though the tensor has a total number of elements a multiple of 4,
    # its data pointer will be offset by sizeof(float), possibly breaking the float4 load.
    output = cuda_module.forward(misaligned)
    expected = torch.relu(misaligned)
    # We compare the outputs even though the misaligned access might lead to errors or wrong results.
    # If the kernel mishandles misaligned memory, the result will differ from expected.
    assert torch.allclose(output, expected, atol=1e-5), "Misaligned memory access did not produce expected output."

def test_non_contiguous_input(cuda_module):
    # Create a contiguous tensor then take a non-contiguous slice.
    x = torch.randn(64, 128, dtype=torch.float32, device="cuda")
    # Transpose produces a non-contiguous tensor.
    non_contiguous = x.t()
    # The kernel expects contiguous memory. Even though PyTorch may allow non-contiguous tensors,
    # the kernel uses data_ptr assuming contiguous layout.
    with pytest.raises(RuntimeError):
        # If the kernel or PyTorch runtime catches non-contiguity, it should raise an error.
        cuda_module.forward(non_contiguous)

def test_extremely_large_tensor_configuration(cuda_module):
    # Although it is not practical to allocate an extremely large tensor,
    # we simulate the scenario by using a tensor size close to the int32 limit for threads configuration.
    # (Note: This test might be only conceptual or skipped if system memory is insufficient.)
    num_elements = 2**24  # About 16 million elements; adjust if necessary to approach limits.
    x = torch.randn(num_elements, dtype=torch.float32, device="cuda")
    output = cuda_module.forward(x)
    expected = torch.relu(x)
    assert torch.allclose(output, expected, atol=1e-5), "Kernel failed with large tensor configuration."

def test_cpu_input_error(cuda_module):
    # Passing a CPU tensor might not be supported by the kernel.
    x = torch.randn(1024, dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError):
        cuda_module.forward(x)
