
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 & Issue 2: Incorrect thread mapping causes race conditions and wrong results.
# Here we check that the custom kernel result does not match the expected PyTorch result for inputs 
# whose total number of elements is a multiple of 4.
def test_vectorized_race_condition():
    # Use a tensor whose total number of elements is a multiple of 4
    n = 16384  # divisible by 4
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    mod = build_kernel()
    out = mod.forward(x, 0.01)
    expected = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    # Due to the race condition and mis-indexing in the vectorized part,
    # the result should differ from the expected output.
    assert not torch.allclose(out, expected, atol=1e-5), (
        "Race condition/vectorization issue not triggered: "
        "kernel output unexpectedly matches the expected output."
    )

# Issue 1 & Issue 2 (Vectorized vs. remainder handling):
# Use a tensor whose total number of elements is not divisible by 4.
def test_remainder_element_handling():
    n = 16387  # not divisible by 4
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    mod = build_kernel()
    out = mod.forward(x, 0.01)
    expected = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    # The output should be incorrect because of the overlapping processing in the vectorized part.
    assert not torch.allclose(out, expected, atol=1e-5), (
        "Remainder handling issue not triggered: "
        "kernel output unexpectedly matches the expected output."
    )

# Issue 3: The kernel does not verify the input tensor’s dtype.
# Passing a non-float32 tensor (e.g. float64) should cause an error or produce incorrect results.
def test_invalid_dtype():
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        _ = mod.forward(x, 0.01)

# Issue 4: The kernel does not check for launch errors.
# While it is not straightforward to simulate a kernel launch error in a regular test,
# we can at least check that a non-contiguous tensor (which should be rejected by CHECK_CONTIGUOUS) fails.
def test_non_contiguous_tensor():
    x = torch.randn(1, 16384, device="cuda", dtype=torch.float32)
    x_noncontig = x.transpose(0, 1)  # Make it non-contiguous.
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        _ = mod.forward(x_noncontig, 0.01)
