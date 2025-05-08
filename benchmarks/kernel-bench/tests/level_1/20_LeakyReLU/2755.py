
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_leaky_relu",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_module():
    mod = build_kernel()
    yield mod

def test_input_data_type_issue(cuda_module):
    # Issue 1: Passing a float64 tensor should cause problems because the kernel expects float32.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # The kernel does not check the data type; it will reinterpret the data as float,
    # which should produce an incorrect result.
    res = cuda_module.forward(x, 0.01)
    torch.cuda.synchronize()
    expected = F.leaky_relu(x, negative_slope=0.01)
    # We expect the result to not match the correct LeakyReLU output.
    assert not torch.allclose(res, expected, atol=1e-5), (
        "Kernel accepted a float64 tensor and produced a result close to the expected "
        "value, but it should only support float32."
    )

def test_memory_alignment_issue(cuda_module):
    # Issue 2: The kernel assumes the input is 16-byte aligned.
    # Create a larger tensor and then slice it to generate a misaligned tensor.
    base = torch.randn(1024 + 1, device="cuda", dtype=torch.float32)
    x = base[1:]  # Likely misaligned.
    res = cuda_module.forward(x, 0.01)
    torch.cuda.synchronize()
    expected = F.leaky_relu(x, negative_slope=0.01)
    # If misalignment causes incorrect memory accesses, the result will deviate from expected.
    assert not torch.allclose(res, expected, atol=1e-5), (
        "Kernel output for a misaligned tensor should differ from the correct result."
    )

def test_non_divisible_by_four_issue(cuda_module):
    # Issue 3: When the total number of elements is not divisible by 4,
    # the kernel must correctly process the "vectorized" part and the remainder.
    # Create a tensor whose number of elements is not divisible by 4.
    # For example, a 1D tensor of size 1025.
    x = torch.randn(1025, device="cuda", dtype=torch.float32)
    res = cuda_module.forward(x, 0.01)
    torch.cuda.synchronize()
    expected = F.leaky_relu(x, negative_slope=0.01)
    # If the cleanup kernel does not cover the remainder correctly, results will differ.
    assert torch.allclose(res, expected, atol=1e-5), (
        "Kernel did not correctly handle inputs whose sizes are not divisible by 4."
    )

def test_hardcoded_warp_block_configurations_issue(cuda_module):
    # Issue 4: The kernel uses hardcoded warp and block configurations.
    # Test with a large tensor to ensure that the fixed configuration scales.
    x = torch.randn(100000, device="cuda", dtype=torch.float32)
    res = cuda_module.forward(x, 0.01)
    torch.cuda.synchronize()
    expected = F.leaky_relu(x, negative_slope=0.01)
    # If the hardcoded configuration was not general enough, the output may be incorrect.
    assert torch.allclose(res, expected, atol=1e-5), (
        "Kernel failed on a large input, suggesting that the hardcoded warp/block "
        "configuration is not general enough."
    )

def test_missing_cuda_error_checking_issue(cuda_module):
    # Issue 5: The kernels do not check for launch errors.
    # We can simulate an error by passing a CPU tensor, which should trigger the CHECK_CUDA macro.
    x = torch.randn(1024, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        res = cuda_module.forward(x, 0.01)
        torch.cuda.synchronize()
