
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1 & 2: When D is less than warp_size, the shared memory allocation becomes zero 
# and the warp reduction uses a fixed mask. This test tries with D < 32 and expects a failure.
def test_small_D_for_shared_memory_and_warp_reduction():
    cuda_mod = build_kernel()
    # Use D smaller than 32 (e.g., 16) to trigger the shared memory mis-allocation 
    # and improper warp reduction mask usage.
    batch_size = 4
    D = 16  # less than 32 so threads = min(256,16) = 16 -> (16/32)==0 shared mem allocated
    x = torch.randn(batch_size, D, device="cuda", dtype=torch.float32)
    
    # Although behavior is undefined, we expect that the kernel will eventually
    # throw a runtime error or produce NaNs due to lack of shared memory.
    out = cuda_mod.forward(x)
    torch.cuda.synchronize()
    # Check if the output norm is valid: if normalization failed, some rows may be NaN.
    row_norms = torch.sum(torch.abs(out), dim=1)
    if torch.isnan(row_norms).any():
        pytest.fail("Kernel produced NaNs because of insufficient shared memory / improper warp reduction.")
    # Alternatively, if the output norm is not near 1 (it should be for L1-normalization),
    # then we assume the reduction failed.
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-3), \
        "Row L1 norms not close to 1, indicating a warp reduction/shared memory issue."

# Issue 3: The kernel does not check the data type and will cast non-float32 tensors improperly.
def test_wrong_dtype_input():
    cuda_mod = build_kernel()
    batch_size = 4
    D = 128  # pick any dimension where shared memory is allocated correctly
    # Create a double input tensor.
    x = torch.randn(batch_size, D, device="cuda", dtype=torch.float64)
    
    # The kernel uses x.data_ptr<float>(), so if a double tensor is passed,
    # the memory will be misinterpreted. We check that the normalized result
    # does not match correct L1 normalization computed in python.
    with torch.no_grad():
        out_kernel = cuda_mod.forward(x)  # type punning error expected
        torch.cuda.synchronize()
    
    # Compute the true output using PyTorch operations.
    norm_val = torch.sum(torch.abs(x), dim=1, keepdim=True).clamp_min(1e-12)
    out_ref = x / norm_val

    # Because the kernel treats the underlying bytes as float32,
    # the output will be numerically very different.
    if torch.allclose(out_kernel, out_ref, atol=1e-5):
        pytest.fail("Kernel unexpectedly produced correct results with a double input. "
                    "It should only support float32.")
    else:
        # If results differ, we consider the issue triggered.
        assert True
