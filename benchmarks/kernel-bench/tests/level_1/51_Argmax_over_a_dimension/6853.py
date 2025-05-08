
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to compile and load the CUDA kernel from "kernel.cu"
def build_kernel():
    module = load(
        name="argmax_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return module

# Test 1: Using a tensor with the wrong dtype should trigger the float32 check.
def test_wrong_dtype():
    # Create a double tensor. The kernel only accepts float32.
    x = torch.randn(16, 256, 256, dtype=torch.float64, device='cuda')
    kernel_mod = build_kernel()
    with pytest.raises(RuntimeError, match="Only float32 is supported."):
        # The kernel call is wrapped inside our binding "forward" call.
        kernel_mod.forward(x, 1)
        
# Test 2: An input where the reduction dimension is empty.
def test_empty_reduction_dim():
    # Create a tensor with reduction dimension of size 0.
    # For example, shape: (batch, reduction, inner)
    x = torch.randn(8, 0, 10, dtype=torch.float32, device='cuda')
    kernel_mod = build_kernel()
    # PyTorch's native torch.argmax() will error on empty dims.
    with pytest.raises(RuntimeError):
        # Our custom kernel is not handling dimSize = 0 so it should produce an error.
        kernel_mod.forward(x, 1)

# Test 3: A case to trigger potential integer overflow in computing sizes.
# NOTE: Allocating a tensor that actually overflows int32 arithmetic is impractical,
# so we simulate the situation by using tensor shapes that are huge.
# Even if the device can not really allocate such a tensor, the goal is to trigger the overflow path.
@pytest.mark.skip(reason="This test is designed to simulate an overflow scenario on huge tensor sizes. Allocation might be impossible on the current device.")
def test_integer_overflow():
    # We choose dimensions such that outerSize * innerSize exceeds the maximum 32-bit positive integer.
    # For instance, let input shape be (N, 3, M)
    # outerSize = N, dimSize = 3, innerSize = M, and we want N * M > 2^31-1.
    # One possible choice: N = 50000, M = 50000 (50000*50000 = 2.5e9 > 2^31-1)
    N = 50000
    M = 50000
    # This tensor is huge in memory so normally allocation will fail.
    # We attempt to allocate a small tensor and then "simulate" the huge sizes via a monkey-patch.
    x = torch.randn(1, 3, 1, dtype=torch.float32, device='cuda')

    # Monkey-patch the 'sizes' (and shape) of x to emulate a huge tensor.
    # WARNING: This is purely artificial; the underlying data storage is not big enough.
    # However, it will force the kernel to compute wrong base offsets due to overflowing int arithmetic.
    fake_shape = (N, 3, M)
    # note: We cannot directly modify x.shape, so we create a fake tensor class.
    class FakeTensor:
        def __init__(self, tensor, fake_shape):
            self.tensor = tensor
            self.fake_shape = fake_shape
        def contiguous(self):
            return self
        def sizes(self):
            return self.fake_shape
        def dim(self):
            return len(self.fake_shape)
        def device(self):
            return self.tensor.device
        def data_ptr(self):
            return self.tensor.data_ptr()
        def __getattr__(self, attr):
            # Forward other attribute access to the underlying tensor.
            return getattr(self.tensor, attr)
    fake_x = FakeTensor(x, fake_shape)

    kernel_mod = build_kernel()
    # We run the kernel. We do not know what the kernel will output, but if overflow occurs,
    # the result will be different from torch.argmax (which runs in C++ with 64-bit indices).
    with pytest.raises(Exception):
        # Expect an error (or an incorrect computation that we can check)
        indices = kernel_mod.forward(fake_x, 1)
        # Optionally, we could compare with torch.argmax on a small valid tensor
        # but here, we expect the kernel to misbehave due to overflow.
        torch.cuda.synchronize()
