
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="softsign_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Test case 1: Trigger type-related issue
# Passing a tensor of type double. The CUDA kernel will reinterpret the data as float,
# and the output will not match the expected softsign activation.
def test_wrong_dtype():
    softsign = build_kernel()
    # Create a tensor with dtype double on CUDA
    x = torch.randn(1024, dtype=torch.double, device="cuda")
    # Expected softsign (computed in torch, using double precision)
    expected = x / (1 + torch.abs(x))
    
    # Call the CUDA kernel - note that the kernel expects float data
    out = softsign.forward(x)
    # Since data is misinterpreted, the result will diverge from expected.
    # We assert that the output is not close.
    assert not torch.allclose(out, expected, atol=1e-5), "Kernel unexpectedly handled double-precision input correctly!"

# Test case 2: Trigger non-contiguous tensor issue
# Passing a non-contiguous tensor should cause the CHECK_CONTIGUOUS macro to fail.
def test_non_contiguous_input():
    softsign = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous by transposing it
    x = torch.randn(64, 16, dtype=torch.float32, device="cuda").t()  # Transpose makes it non-contiguous
    with pytest.raises(RuntimeError, match=r".*must be contiguous.*"):
        softsign.forward(x)

# Test case 3: (Optional) Trigger stream creation/launch error indirectly.
# One potential way is to pass an empty tensor. Although not directly an error in stream creation,
# it might trigger unexpected behavior if the kernel or stream logic does not handle zero-sized inputs.
def test_empty_input():
    softsign = build_kernel()
    x = torch.empty(0, dtype=torch.float32, device="cuda")
    # For an empty tensor, the output should also be empty.
    out = softsign.forward(x)
    assert out.numel() == 0, "Kernel failed to handle an empty input tensor!"

if __name__ == "__main__":
    pytest.main([__file__])
