
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the kernel extension from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    # Issue 1: The kernel assumes the input tensor is contiguous.
    # Create a contiguous tensor and then make a non-contiguous view (by transposition).
    # For example, for a tensor of shape (batch_size, dim1, dim2), transposing
    # two dimensions will result in a non‐contiguous tensor.
    batch_size = 16
    dim1 = 256
    dim2 = 256
    # Original contiguous tensor
    x = torch.randn(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    # Create a non-contiguous view (swap the last two dimensions)
    x_noncontig = x.transpose(1, 2)
    reduce_dim = 2  # Note: after transpose, reduction dimension changes

    my_module = build_kernel()
    # Compute with custom CUDA kernel
    out_kernel = my_module.forward(x_noncontig, reduce_dim)
    # Compute expected result using PyTorch (which works correctly for non-contiguous tensors)
    out_ref = torch.sum(x_noncontig, dim=reduce_dim, keepdim=True)
    # The kernel, however, assumes a specific memory layout and may produce an incorrect result.
    # We expect the result to differ.
    if torch.allclose(out_kernel, out_ref, atol=1e-4):
        raise AssertionError("Non-contiguous input did not trigger an error in the kernel as expected.")

@pytest.mark.skip(reason="This test simulates a large reduction dimension to trigger potential integer overflow. "
                         "Allocating such a large tensor is impractical; this test uses a monkey-patch simulation.")
def test_large_reduce_dimension(monkeypatch):
    # Issue 2: The loop counter is a 32-bit int while reduce_size is int64_t.
    # For very large reduction dimensions, this can lead to overflow.
    # We cannot really allocate a tensor with reduce_size > INT_MAX;
    # instead, we simulate the condition by forcing the sizes computed inside the wrapper.
    
    # First, build a reasonably sized tensor on GPU.
    x = torch.randn(2, 1024, 2, device='cuda', dtype=torch.float32)
    # Monkey-patch the sizes vector returned by input.sizes().vec() to simulate a huge reduce_size.
    fake_sizes = list(x.sizes())
    # Suppose we want the reduction dimension (say, dim=1) to appear huge.
    fake_sizes[1] = 2**31  # a value exceeding 32-bit integer range
    
    # Patch the torch.size() method for our tensor.
    original_sizes = x.sizes
    x_sizes_called = False
    def fake_sizes_func():
        nonlocal x_sizes_called
        x_sizes_called = True
        return torch.Size(fake_sizes)
    monkeypatch.setattr(x, "sizes", fake_sizes_func)
    
    my_module = build_kernel()
    
    # When calling the kernel, the computed reduce_size will be huge (fake_sizes[1]),
    # and the loop counter in the kernel (declared as int) may overflow.
    # We compare against the expected result computed with torch.sum.
    # Note: This test is only a simulation: we expect the kernel to produce an incorrect result.
    try:
        out_kernel = my_module.forward(x, 1)
        # Compute expected result using PyTorch (but note: torch.sum will use the original x sizes,
        # so we simulate the issue by applying torch.sum with the fake reduce dimension).
        # In practice the test should detect that the kernel result is not equal to the reference.
        out_ref = torch.sum(x, dim=1, keepdim=True)
        if torch.allclose(out_kernel, out_ref, atol=1e-4):
            raise AssertionError("Large reduction dimension did not trigger the integer overflow issue as expected.")
    except RuntimeError as e:
        # If the kernel launch fails with an error referencing index issues, then the test passes.
        assert "index" in str(e) or "overflow" in str(e)
    finally:
        # Restore the original sizes method
        monkeypatch.setattr(x, "sizes", original_sizes)
