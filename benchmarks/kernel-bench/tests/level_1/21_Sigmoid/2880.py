
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build and load our custom CUDA extension
def build_kernel():
    cuda_module = load(
        name="custom_sigmoid",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# 1. Test that the module does not catch kernel launch errors:
# Since the kernel code itself does not do error checking,
# we simulate an error by passing an obviously incorrect size.
# The test will check for a CUDA error (e.g. kernel launch failure).
def test_kernel_launch_error():
    my_module = build_kernel()
    # Create a tensor on device
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    # Intentionally pass an invalid tensor size by creating a zero-size tensor.
    # Although the kernel does not check errors explicitly, a bad launch might
    # result in an error after device synchronization.
    x_bad = x.narrow(1, 0, 0)  # zero columns, numel == 0
    with pytest.raises(RuntimeError):
        # The kernel launch should error out or produce an error when synchronizing,
        # because launching a kernel with zero threads (or an invalid size) is problematic.
        out = my_module.forward(x_bad)
        torch.cuda.synchronize()

# 2. Test misaligned memory access.
# For the vectorized kernel, the pointer is reinterpreted as float4.
# We force misalignment by taking a non-contiguous slice.
def test_vectorized_misaligned_memory():
    my_module = build_kernel()
    # Create a tensor with an extra element so that we can slice off a misaligned view 
    x_full = torch.randn(1025, device="cuda", dtype=torch.float32)
    # Since float4 requires alignment of 4 elements, take a slice that shifts the pointer by 1 element.
    x = x_full[1:].clone()  # Clone to force a new allocation; likely misaligned.
    # Run our custom kernel
    out_custom = my_module.forward(x)
    # Compare to PyTorch's built-in sigmoid; they should be nearly equal.
    out_ref = torch.sigmoid(x)
    assert torch.allclose(out_custom, out_ref, atol=1e-5), "Mismatch due to misaligned memory handling."

# 3. Test handling of double precision.
# The scalar kernel always casts to float and uses expf(), so results for double inputs
# may suffer precision loss.
def test_double_precision_input():
    my_module = build_kernel()
    x = torch.randn(1024, device="cuda", dtype=torch.double)
    out_custom = my_module.forward(x)
    out_ref = torch.sigmoid(x)
    # Use a looser tolerance for double since computation is done in float.
    assert torch.allclose(out_custom, out_ref, atol=1e-4), "Double precision input did not match reference."

# 4. Test half precision execution.
# The kernel does not provide a vectorized implementation for half precision and falls back to the scalar kernel.
def test_half_precision_input():
    my_module = build_kernel()
    x = torch.randn(2048, device="cuda", dtype=torch.half)
    out_custom = my_module.forward(x)
    out_ref = torch.sigmoid(x)
    # Half precision often needs a higher relative tolerance.
    assert torch.allclose(out_custom.float(), out_ref.float(), atol=1e-3), "Half precision input did not match reference."

# 5. Test consistency between vectorized and scalar kernels.
# Provide two inputs – one where number of elements is divisible by 4 (vectorized path)
# and one where it is not (scalar tail path included) – and compare with PyTorch's sigmoid.
def test_vectorized_vs_scalar_consistency():
    my_module = build_kernel()
    # Input size divisible by 4 (vectorized exclusively)
    x_div = torch.randn(1024, device="cuda", dtype=torch.float32)
    out_div = my_module.forward(x_div)
    ref_div = torch.sigmoid(x_div)
    assert torch.allclose(out_div, ref_div, atol=1e-5), "Vectorized kernel output does not match reference for size divisible by 4."

    # Input size not divisible by 4 (ensuring tail elements processed by scalar code)
    x_nondiv = torch.randn(1023, device="cuda", dtype=torch.float32)
    out_nondiv = my_module.forward(x_nondiv)
    ref_nondiv = torch.sigmoid(x_nondiv)
    assert torch.allclose(out_nondiv, ref_nondiv, atol=1e-5), "Kernel output does not match reference for size not divisible by 4."
