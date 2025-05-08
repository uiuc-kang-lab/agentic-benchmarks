
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="mse_kernel",
        sources=["kernel.cu"],  # kernel file
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# 1. Test for unsupported input type: half-precision.
def test_unsupported_half_precision():
    # Create half-precision CUDA tensors.
    preds = torch.randn(128, 4096, device="cuda").half()
    tgts = torch.randn(128, 4096, device="cuda").half()
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError, match="AT_DISPATCH"):
        # Expect the dispatch macro to fail due to unsupported type.
        my_module.forward(preds, tgts)

# 2. Test for non-contiguous input tensors.
def test_non_contiguous_inputs():
    # Create a contiguous tensor and then generate a non-contiguous version.
    base = torch.randn(128, 4096, device="cuda")
    # For example, transpose a 2D tensor to make it non-contiguous.
    preds = base.t().contiguous()[::2].t()  # artificially break contiguity
    tgts = base.t().contiguous()[::2].t()   # similarly for targets

    # Even though the element counts match,
    # the kernel uses .data_ptr() and assumes contiguous layout.
    my_module = build_kernel()
    # Run the kernel: this may produce an incorrect result when non-contiguous
    out = my_module.forward(preds, tgts)
    # Compute the expected result using a simple PyTorch MSE (which handles non-contiguity)
    mse_expected = torch.mean((preds - tgts) ** 2)
    # Likely the results will not match because of non-contiguous memory accesses.
    assert not torch.allclose(out, mse_expected, atol=1e-5), \
        "Kernel produced the correct result for non-contiguous inputs unexpectedly."

# 3. Test for unsupported input type: integer tensors.
def test_unsupported_integer_dtype():
    # Create integer tensors on CUDA.
    preds = torch.randint(0, 10, (128, 4096), device="cuda", dtype=torch.int32)
    tgts = torch.randint(0, 10, (128, 4096), device="cuda", dtype=torch.int32)
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError, match="Input sizes must match"):
        # Even though the sizes match, the dispatch macros for AT_DISPATCH_FLOATING_TYPES
        # should not match integer types, leading to an error.
        my_module.forward(preds, tgts)

# 4. Test for detecting lack of kernel launch error checking.
def test_kernel_launch_error_detection():
    # In our current implementation, if we supply tensors with mismatched number of elements,
    # the TORCH_CHECK should trigger an error. This test case exercises that.
    preds = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    # Create a target tensor that is intentionally of a different size.
    tgts = torch.randn(128, 4095, device="cuda", dtype=torch.float32)
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError, match="Input sizes must match"):
        my_module.forward(preds, tgts)
