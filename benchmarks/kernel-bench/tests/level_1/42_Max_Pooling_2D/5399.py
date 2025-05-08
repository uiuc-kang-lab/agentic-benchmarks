
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Builds the extension from kernel.cu. Adjust the extra flags if needed.
    return load(
        name="maxpool2d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Test using an integer tensor.
def test_integer_tensor():
    # Create an integer tensor. The kernel uses numeric_limits<...>::infinity()
    # which is not defined for ints.
    x = torch.randint(low=0, high=255, size=(8, 3, 16, 16), dtype=torch.int32, device="cuda")
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel launch to fail because the dispatch macro does not
        # handle integer types.
        _ = module.forward(x, 2, 2, 0, 1)

# Issue 2: Test using half-precision (float16) tensor.
def test_half_precision_tensor():
    # Create a half precision tensor. The dispatch macro used in the extension
    # does not support half precision.
    x = torch.randn(8, 3, 16, 16, device="cuda", dtype=torch.float16)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel launch to raise an error since half is not dispatched.
        _ = module.forward(x, 2, 2, 0, 1)

# Issue 3: Test that exposes the ambiguity in the use of max.
#
# Here we use a double precision tensor in which the use of std::numeric_limits<scalar_t>::infinity()
# and an unqualified max may compile but lead to precision/semantics issues in a complex kernel.
def test_double_precision_max_behavior():
    # Create a double precision tensor.
    x = torch.randn(8, 3, 16, 16, device="cuda", dtype=torch.float64)
    module = build_kernel()
    y = module.forward(x, 2, 2, 0, 1)
    # For a simple maxpool2d, compare against PyTorch's own implementation.
    ref = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0, dilation=1)
    # The outputs may differ if the max function in the kernel does not behave correctly.
    # This test expects that the max is computed with the correct semantics.
    assert torch.allclose(y, ref), f"Kernel max behavior is unexpected, difference: {(y-ref).abs().max()}"

# Issue 4: Test using a non-contiguous input tensor.
def test_non_contiguous_input():
    # Create a contiguous tensor then make it non-contiguous by transposing.
    x = torch.randn(8, 3, 16, 16, device="cuda")
    # Transpose to make non-contiguous. Note: the underlying pooling expects 
    # input shape to be (N,C,H,W) in contiguous order.
    x_noncontig = x.transpose(1, 2)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # We expect that the kernel which performs raw pointer arithmetic on contiguous
        # data will not work correctly on a non-contiguous tensor.
        _ = module.forward(x_noncontig, 2, 2, 0, 1)
