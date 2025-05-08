
import torch
import pytest
from torch.utils.cpp_extension import load

# Function to build the CUDA extension module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="argmin_cuda_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return build_kernel()

# Test case 1: Non-contiguous input tensor
def test_non_contiguous_input(cuda_module):
    # Create a contiguous tensor and then make a non-contiguous version by transposing
    x = torch.randn(16, 256, 256, device="cuda")
    x_noncontig = x.transpose(1, 2)  # now non-contiguous with shape (16,256,256)
    # Expected result computed by torch.argmin along dimension 1 (if we assume user passes dim=1)
    expected = torch.argmin(x_noncontig, dim=1)
    # Using the CUDA kernel: our kernel expects the reduction dimension to be contiguous.
    # Here we deliberately pass a non-contiguous tensor.
    output = cuda_module.forward(x_noncontig, 1)
    torch.cuda.synchronize()
    # The output is likely to be wrong because the kernel does not account for non-contiguity.
    assert not torch.allclose(output, expected), "Kernel unexpectedly handled non-contiguous tensors correctly."

# Test case 2: Concurrent kernel invocations using different streams
def test_concurrent_invocations(cuda_module):
    # This test is designed to stress the use of global constant memory.
    # Two different tensors with different shape parameters are processed concurrently.
    # For one invocation, reduction dimension is different from the other.
    x1 = torch.randn(8, 64, 32, device="cuda")
    x2 = torch.randn(8, 32, 64, device="cuda")  # different size along reduction dimension

    # Expected outputs using torch.argmin
    expected1 = torch.argmin(x1, dim=1)
    expected2 = torch.argmin(x2, dim=1)

    # Use different CUDA streams and launch kernel concurrently.
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    output1 = torch.empty(expected1.shape, device="cuda", dtype=torch.long)
    output2 = torch.empty(expected2.shape, device="cuda", dtype=torch.long)

    # Launch the kernels on different streams by setting the current stream.
    with torch.cuda.stream(stream1):
        tmp1 = cuda_module.forward(x1, 1)
    with torch.cuda.stream(stream2):
        tmp2 = cuda_module.forward(x2, 1)

    # Synchronize streams.
    stream1.synchronize()
    stream2.synchronize()

    # Because __constant__ memory is global and shared, the parameters from the second
    # launch might overwrite those from the first, producing incorrect results.
    # We expect at least one of the outputs to differ from the CPU computed argmin.
    correct1 = torch.allclose(tmp1, expected1)
    correct2 = torch.allclose(tmp2, expected2)
    assert not (correct1 and correct2), ("Concurrent invocations did not trigger an error as expected. "
                                          "At least one result should be wrong due to race conditions with "
                                          "global constant memory use.")
