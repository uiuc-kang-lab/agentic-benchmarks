
import pytest
import torch
import math
from torch.utils.cpp_extension import load

# Helper to compile and load the CUDA extension.
def build_kernel():
    kernel = load(
        name="softplus_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return kernel

# 1. Test with input size not a multiple of 512 (hard-coded block size issue).
def test_non_multiple_of_block_size():
    # Create an input tensor whose total number of elements is not a multiple of 512.
    # The kernel uses "if (idx < size)" so it should still be functionally correct,
    # but this test ensures that the kernel works correctly for any size.
    kernel = build_kernel()
    # Use a tensor with an odd number of elements.
    N = 1024 + 37  # Not a multiple of 512.
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    out_cuda = kernel.forward(x)
    # Reference using PyTorch's built-in softplus.
    out_ref = torch.nn.functional.softplus(x)
    torch.cuda.synchronize()
    assert torch.allclose(out_cuda, out_ref, atol=1e-5), \
        f"Output differs for tensor size not multiple of block size. Max diff: {(out_cuda - out_ref).abs().max()}"

# 2. Test with half precision to trigger lack of support.
def test_half_precision_not_supported():
    kernel = build_kernel()
    # Create a half precision tensor.
    x = torch.randn(1000, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # Expect a runtime error because float16 is not handled by AT_DISPATCH_FLOATING_TYPES.
        _ = kernel.forward(x)

# 3. Test potential precision issue from using double literals:
# Create an input tensor with values close to the threshold boundaries.
def test_threshold_precision():
    kernel = build_kernel()
    # Create tensor values exactly at, just above, and just below 20 and -20.
    vals = torch.tensor([-20.0, -20.001, -19.999, 20.0, 20.001, 19.999],
                        device="cuda", dtype=torch.float32)
    out_cuda = kernel.forward(vals)
    out_ref = torch.nn.functional.softplus(vals)
    torch.cuda.synchronize()
    # We use a looser tolerance since the threshold branch may introduce slight differences.
    assert torch.allclose(out_cuda, out_ref, atol=1e-4), \
        f"Threshold precision issue: CUDA output {out_cuda.cpu().numpy()} doesn't match reference {out_ref.cpu().numpy()}"

# 4. Test lack of error checking by triggering a launch error.
# One way to force an error is to pass a tensor with an integer type.
def test_input_type_error():
    kernel = build_kernel()
    x_int = torch.randint(0, 10, (1000,), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _ = kernel.forward(x_int)

# 5. Test non-contiguous tensor.
def test_non_contiguous_tensor():
    kernel = build_kernel()
    # Create a 2D tensor and then take a non-contiguous slice (transpose makes it non-contiguous).
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    # Transpose creates a non-contiguous view.
    x_noncontig = x.t()
    # Compute the softplus on the non-contiguous tensor using the CUDA kernel.
    out_cuda = kernel.forward(x_noncontig)
    # Compute reference result using PyTorch built-in softplus.
    out_ref = torch.nn.functional.softplus(x_noncontig)
    torch.cuda.synchronize()
    # The kernel assumes contiguous memory, so the result is likely incorrect.
    # Therefore, we expect a significant difference.
    if torch.allclose(out_cuda, out_ref, atol=1e-5):
        pytest.fail("Kernel did not fail or produce a wrong result for non-contiguous input as expected.")

if __name__ == '__main__':
    pytest.main([__file__])
