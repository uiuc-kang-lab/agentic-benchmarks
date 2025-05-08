
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="selu_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Pass a half precision tensor to trigger the type limitation.
def test_half_precision():
    my_module = build_kernel()
    # Create a half precision tensor on CUDA.
    x = torch.randn(1024, device="cuda", dtype=torch.half)
    with pytest.raises(RuntimeError):
        # Dispatch should fail because AT_DISPATCH_FLOATING_TYPES does not cover half.
        my_module.forward(x)
        
# Test 2: Pass a non-contiguous tensor to see that the kernel does not handle it properly.
def test_non_contiguous():
    my_module = build_kernel()
    # Create a contiguous tensor first.
    x = torch.randn(16, 32, 8, device="cuda", dtype=torch.float32)
    # Make it non-contiguous by permuting dimensions.
    x_noncontig = x.permute(2, 0, 1)
    # The kernel expects contiguous memory, so its output will differ from torch.selu.
    out_kernel = my_module.forward(x_noncontig)
    # Compute expected result using torch.selu on the same non-contiguous tensor.
    out_ref = torch.selu(x_noncontig)
    # Since underlying memory layout is different, the kernel may yield wrong results.
    # We require that the results are NOT close to trigger the issue.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), \
        "Kernel unexpectedly handled non-contiguous input correctly."

# Test 3: Pass a CPU tensor which should trigger the TORCH_CHECK in the kernel.
def test_cpu_input():
    my_module = build_kernel()
    x_cpu = torch.randn(1024, dtype=torch.float32)  # CPU tensor
    with pytest.raises(RuntimeError):
        my_module.forward(x_cpu)
        
# Additional note: While we cannot easily test for the unused shared memory or the
# lack of kernel launch error checking from Python, the above tests help trigger
# failures specific to type and memory layout issues.
