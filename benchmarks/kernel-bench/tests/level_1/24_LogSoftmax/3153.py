
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper: Build the kernel module from kernel.cu.
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


# 1. Test for mis‐aligned shared memory usage.
#    We “stress” the kernel by passing double precision input.
#    (On some systems this mis‐alignment may cause the result to be wrong,
#    so we compare against PyTorch's built-in log_softmax.)
def test_shared_memory_alignment_issue():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    kernel_module = build_kernel()

    batch_size = 4
    dim = 1024  # sufficiently large so that several warps and shared mem are used.
    # Use double data (float64) for stronger alignment requirements.
    x = torch.randn(batch_size, dim, dtype=torch.float64, device="cuda")
    # Run our CUDA kernel
    y_kernel = kernel_module.forward(x, 1)
    # Run native PyTorch log_softmax as reference.
    y_ref = torch.log_softmax(x, dim=1)
    # In case of mis-alignment one might get very different numbers.
    assert torch.allclose(y_kernel, y_ref, atol=1e-5), \
        f"Output does not match reference; possible shared memory alignment issue."


# 2. Test for the case where the reduction dimension is empty.
#    In an empty reduction (dim_size==0) the kernel will not perform any writes,
#    so the output will likely be uninitialized.
def test_empty_reduction_dimension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    kernel_module = build_kernel()

    batch_size = 4
    dim = 0  # empty reduction dimension
    # Create an input with a zero-sized softmax dimension.
    x = torch.randn(batch_size, dim, device="cuda")
    # Running torch.log_softmax on an empty dimension returns an empty tensor.
    y_ref = torch.log_softmax(x, dim=1)
    # Run our CUDA kernel.
    # This call is expected to result in different output (or even undefined values)
    # because the kernel doesn’t check for the empty dim_size.
    y_kernel = kernel_module.forward(x, 1)
    # We check that the shapes match but that the values are not correct.
    # (If the kernel were fixed the outputs should match the reference.)
    assert y_kernel.shape == y_ref.shape, "Output shape mismatch for empty dim."
    # Compare each element – if the kernel simply leaves memory uninitialized,
    # the results will differ from torch.log_softmax (which returns an empty tensor).
    # If they match exactly then there is no error.
    assert not torch.allclose(y_kernel, y_ref), \
        "Kernel output unexpectedly matches the reference in the empty-dim case."


# 3. Test for lack of kernel launch error checking:
#    Here we intentionally pass an invalid dimension (out-of-range) to trigger the TORCH_CHECK.
def test_invalid_dimension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    kernel_module = build_kernel()

    batch_size = 4
    dim = 1024
    # create a simple valid input
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32)
    # Pass an invalid dimension (e.g. dim==5 when input.ndim==2) to trigger error checking.
    with pytest.raises(RuntimeError):
        _ = kernel_module.forward(x, 5)
