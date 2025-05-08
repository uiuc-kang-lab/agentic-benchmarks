
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

# Test case for issue 1:
# The kernel does not check for CUDA kernel launch errors. One way such errors might be
# triggered is when the wrong device type is passed. The kernel uses TORCH_CHECK to ensure
# A is a CUDA tensor, so passing a CPU tensor should raise an error. 
def test_wrong_device():
    my_module = build_kernel()
    # Create a CPU tensor (instead of CUDA) which should trigger the check.
    A_cpu = torch.randn(128, 128, dtype=torch.float32)
    s = 2.0
    with pytest.raises(RuntimeError, match="Input tensor A must be a CUDA tensor"):
        my_module.forward(A_cpu, s)

# Test case for issue 2:
# Passing a non-contiguous tensor should lead to wrong results since the kernel assumes
# contiguous layout. We deliberately create a non-contiguous tensor (by transposing) and
# compare the kernel result to the expected multiplication result.
def test_non_contiguous_input():
    my_module = build_kernel()
    A = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    # Create a non-contiguous tensor by transposing.
    A_nc = A.t()
    s = 3.14
    # The expected result computed in PyTorch using multiplication works for any layout.
    expected = A_nc * s
    # Call the CUDA kernel forward.
    out = my_module.forward(A_nc, s)
    # Because the kernel incorrectly assumes contiguous memory, the output may be wrong.
    # We expect the results to diverge from the correct value.
    if torch.allclose(out, expected, atol=1e-5):
        pytest.fail("Kernel produced the correct output for a non-contiguous input, "
                    "but it should have failed to account for non-contiguous memory.")

# Test case for issue 3:
# These tests ensure that if a tensor type other than float32 is provided, the kernel fails.
def test_wrong_dtype():
    my_module = build_kernel()
    # Create a tensor of type double.
    A_double = torch.randn(256, 256, device="cuda", dtype=torch.float64)
    s = 2.5
    with pytest.raises(RuntimeError, match="Input tensor A must be of type float"):
        my_module.forward(A_double, s)

if __name__ == "__main__":
    pytest.main([__file__])
