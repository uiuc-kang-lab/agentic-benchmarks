
import pytest
import torch
from torch.utils.cpp_extension import load

# Build/loading helper for the CUDA kernel extension.
def build_kernel():
    cuda_module = load(
        name="mean_reduce_cuda_mod",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Non-contiguous input.
# Create a tensor and then take a transpose so that its layout is no longer contiguous.
# The kernel, however, assumes a contiguous layout along the reduction dimension.
# Therefore the result of our custom kernel should differ from torch.mean.
def test_non_contiguous_input():
    mod = build_kernel()
    # Create a contiguous tensor of shape [batch_size, dim1, dim2]
    batch_size, dim1, dim2 = 4, 8, 16
    x = torch.randn(batch_size, dim1, dim2, device="cuda", dtype=torch.float32)
    # Make x non-contiguous by transposing dim1 and dim2
    x_noncontig = x.transpose(1, 2)
    # We choose to reduce along dimension 1 in the non-contiguous tensor.
    # We compute the reference output using torch.mean on the non-contiguous tensor.
    ref = torch.mean(x_noncontig, dim=1)
    # Call the CUDA kernel (which internally treats the tensor as if it were in contiguous layout)
    out = mod.forward(x_noncontig, 1)
    # The result is expected to differ from the reference because the kernel miscomputes indices.
    # Here we assert that the maximum absolute difference is larger than a small threshold.
    diff = (out - ref).abs().max().item()
    assert diff > 1e-3, f"Non-contiguous input did not trigger the error (max diff = {diff})"

# Test 2: Half precision input.
# Since the reduction kernel uses AT_DISPATCH_FLOATING_TYPES (which does not include half),
# invoking the kernel on a half precision tensor should trigger an error.
def test_half_precision_input():
    mod = build_kernel()
    batch_size, dim1, dim2 = 4, 16, 16
    x = torch.randn(batch_size, dim1, dim2, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        _ = mod.forward(x, 1)

# Test 3: Large tensor dimensions that force a cast from int64_t to int.
# The kernel computes indices using int and thus can overflow when the reduction dimension (or inner size)
# is extremely large. Since it is not practical to allocate enormous tensors in a test, we simulate the
# situation by creating a tensor whose inner dimension is set to a value near the int limit using a
# dummy stride value. Note: This is a synthetic test and may be skipped on boards without sufficient memory.
@pytest.mark.xfail(reason="Simulated large-dimension overflow issue")
def test_large_dimension_overflow():
    mod = build_kernel()
    # We simulate a tensor with a huge inner_size. We use a modest outer size and reduction size.
    # Ideally, inner_size * (reduction_size) should overflow a 32-bit int.
    # Here we choose inner_size close to 2**31; note that allocation of such a tensor is not feasible,
    # so we will simply check that our kernel interface is not prepared to handle such values.
    huge_inner = 2**31  # This value is huge and normally cannot be allocated.
    reduction_size = 16
    outer_size = 2
    # Create a fake tensor with the expected shape [outer_size, reduction_size, huge_inner]
    # We cannot actually allocate it, so we simulate by manually constructing sizes.
    # Instead, we warn that calling kernel.forward with these dimensions would lead to overflow.
    with pytest.raises(RuntimeError):
        # Create a small tensor and then pretend it has huge dimensions by manipulating its size.
        x = torch.randn(outer_size, reduction_size, 16, device="cuda", dtype=torch.float32)
        # Trick: manually set the tensor's shape metadata (this is normally not allowed; so this test is illustrative)
        x_fake = x.as_strided((outer_size, reduction_size, huge_inner),
                              (x.stride(0), x.stride(1), x.stride(2)))
        _ = mod.forward(x_fake, 1)
