
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="l2norm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_multidim_input_issue():
    # This test feeds a 3D tensor to trigger the indexing issue.
    # The kernel expects the normalized dimension to be at index 1 with a corresponding 2D layout.
    input_tensor = torch.randn(4, 16, 8, device="cuda", dtype=torch.float32)
    module = build_kernel()
    output = module.forward(input_tensor)
    # Compute a reference normalization along dim=1 (as intended), but note that the kernel
    # uses a simplified indexing that would likely give a wrong result for 3D inputs.
    ref_norm = input_tensor.norm(p=2, dim=1, keepdim=True) + 1e-12
    reference = input_tensor / ref_norm
    # We expect the output to deviate from the reference due to bad handling of multi-dimensional layouts.
    assert not torch.allclose(output, reference, atol=1e-5), \
        "Expected kernel output to differ from reference for multi-dimensional inputs."

def test_shared_memory_reduction_issue():
    # This test creates an input vector wide enough (C > SEGMENT_SIZE) so multiple blocks per vector are launched.
    # If the shared memory reduction is faulty then the norm will be computed incorrectly.
    C = 1024  # greater than SEGMENT_SIZE (512)
    input_tensor = torch.ones(1, C, device="cuda", dtype=torch.float32)
    module = build_kernel()
    output = module.forward(input_tensor)
    # For an all-ones vector the L2 norm should be sqrt(C) and normalization should yield 1/sqrt(C)
    expected_value = 1.0 / (C ** 0.5)
    # Because of the buggy reduction we expect the output elements to be off from the correct value.
    max_diff = (output - expected_value).abs().max().item()
    assert max_diff > 1e-3, \
        f"Shared memory reduction issue not triggered as expected: max diff = {max_diff}"

def test_double_precision_atomic_issue():
    # This test uses a double precision (float64) tensor.
    # If the hardware or CUDA version does not support atomicAdd for doubles, an error should be triggered.
    C = 1024
    input_tensor = torch.randn(1, C, device="cuda", dtype=torch.float64)
    module = build_kernel()
    with pytest.raises(RuntimeError, match="atomicAdd"):
        _ = module.forward(input_tensor)
        
def test_missing_error_check_issue():
    # Although not a kernel bug per se, the lack of proper error checking after kernel launches may hide synchronization errors.
    # We provide an input whose size makes its processing more challenging.
    C = 256
    input_tensor = torch.randn(1, C, device="cuda", dtype=torch.float32)
    module = build_kernel()
    output = module.forward(input_tensor)
    # Compute correct normalization.
    ref_norm = input_tensor.norm(p=2, dim=1, keepdim=True) + 1e-12
    reference = input_tensor / ref_norm
    # If there was an execution or synchronization issue, we expect the output to differ from the reference.
    assert not torch.allclose(output, reference, atol=1e-5), \
        "Kernel appears to be handling errors silently; expected deviation due to missing error checks."

