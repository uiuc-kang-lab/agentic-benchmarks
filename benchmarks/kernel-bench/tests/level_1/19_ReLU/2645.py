
import os
import pytest
import torch
from torch.utils.cpp_extension import load

# Build/load the CUDA extension from kernel.cu.
def build_kernel():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    module = load(
        name="custom_relu",
        sources=[os.path.join(this_dir, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: Test non-contiguous input handling.
def test_non_contiguous_input():
    # Create a 2-D tensor and then transpose it to make it non-contiguous.
    x = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # now non-contiguous
    kernel = build_kernel()
    try:
        out = kernel.forward(x_noncontig)
    except RuntimeError as e:
        pytest.skip("Kernel does not support non-contiguous inputs: " + str(e))
    # Compare to torch.relu applied to a contiguous tensor.
    out_ref = torch.relu(x_noncontig)
    # If the kernel did not convert the non-contiguous input, the result might be wrong.
    assert torch.allclose(out, out_ref), "Kernel produced incorrect result for non-contiguous input."

# Issue 2: Test half-precision (float16) input.
def test_half_precision_input():
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # The AT_DISPATCH_FLOATING_TYPES macro used in the kernel does not include half.
        _ = kernel.forward(x)

# Issue 3: Test for absence of kernel launch error checking.
# One way to trigger an error is to pass a CPU tensor to a CUDA kernel.
def test_incorrect_device_input():
    x = torch.randn(1024, device="cpu", dtype=torch.float32)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # Because the kernel expects a CUDA tensor and there is no device check inside the kernel,
        # launching with a CPU tensor should result in an error.
        _ = kernel.forward(x)
        
if __name__ == "__main__":
    pytest.main([__file__])
