
import pytest
import torch
from torch.utils.cpp_extension import load
import numpy as np

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 & 2: Race condition and incorrect reduction algorithm.
# This test creates an input with known products so that the computed product is predictable.
# If the shared memory reduction (or missing __syncthreads) is faulty, the kernel’s output may differ.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_incorrect_reduction():
    kernel = build_kernel()
    # Create a tensor of shape (batch, reduce, other) where reduction is along dim=1.
    # We fill the reduction dimension with a known value so that the product can be computed exactly.
    batch = 128
    reduce_dim = 128
    other_dim = 4
    # Use constant value 1.0001 for the reduction dimension so that the product is (1.0001)^(reduce_dim)
    # (Using a slight offset ensures that any extra erroneous multiplications will be noticeable.)
    value = 1.0001
    x = torch.full((batch, reduce_dim, other_dim), value, dtype=torch.float32, device="cuda")
    
    # Compute expected product using torch.prod
    expected = torch.prod(x, dim=1)
    
    # Run the custom CUDA kernel through our module repeatedly to try to catch nondeterministic errors.
    for _ in range(10):
        out = kernel.forward(x, 1)
        # Allow a loose tolerance because floating point error might accumulate further if extra multiplications occur.
        if not torch.allclose(out, expected, atol=1e-4):
            pytest.fail(f"Kernel reduction result incorrect.\nExpected:\n{expected.cpu().numpy()}\nGot:\n{out.cpu().numpy()}")

# Issue 3: Kernel only supports float32.
# This test passes a double tensor to the CUDA kernel and expects either an error or an incorrect result.
# We check if the kernel output does NOT match the torch.prod result computed in double precision.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_input_tensor_type():
    kernel = build_kernel()
    # Create a double tensor input.
    batch = 16
    reduce_dim = 32
    other_dim = 8
    x = torch.randn(batch, reduce_dim, other_dim, dtype=torch.double, device="cuda")
    
    # Even though torch.prod produces a correct result in double, our kernel will try to
    # interpret the memory as float (32-bit) values.
    expected = torch.prod(x, dim=1).to(torch.float32)
    
    # Convert x to contiguous double but our kernel does not support double.
    try:
        out = kernel.forward(x, 1)
    except RuntimeError as e:
        # If an error is raised, that is acceptable, since the kernel is not designed for double.
        return
    # If no error is raised, we check that the result is off by a significant margin.
    # (A proper float interpretation of double data is unlikely to match the expected result.)
    if torch.allclose(out, expected, atol=1e-3):
        pytest.fail("Kernel accepted a double tensor input and produced a result close to double precision torch.prod, but it should only support float32.")

# Optional additional test: Ensure that non‐contiguous tensors are rejected.
# Even though PyTorch already checks for contiguity via CHECK_INPUT, we include this test for completeness.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_contiguous_input():
    kernel = build_kernel()
    # Create a contiguous tensor and then transpose it so that it becomes non‐contiguous.
    x = torch.randn(32, 64, 16, dtype=torch.float32, device="cuda")
    non_contig = x.transpose(0, 1)  # now non-contiguous
    with pytest.raises(Exception):
        kernel.forward(non_contig, 1)
