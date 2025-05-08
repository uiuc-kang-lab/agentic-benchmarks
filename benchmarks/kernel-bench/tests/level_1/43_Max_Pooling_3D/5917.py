
import threading
import pytest
import torch
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="max_pool3d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to call the CUDA kernel wrapper.
def run_pool3d(module, input, kernel_size, stride, padding, dilation, return_indices, ceil_mode):
    return module.forward(input, kernel_size, stride, padding, dilation, return_indices, ceil_mode)

# Test 1: Check API mismatch for return_indices.
# Expecting a tuple (output, indices) but the kernel returns a stacked tensor.
def test_return_indices_api(tmp_path):
    module = build_kernel()
    batch_size, channels, D, H, W = 2, 3, 8, 8, 8
    input = torch.randn(batch_size, channels, D, H, W, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = True
    ceil_mode = False
    
    out = run_pool3d(module, input, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    # We expect a tuple (output, indices) but our kernel stacks these along dim 0.
    # Thus, the first dimension of the result should be 2 instead of the expected 5 dims.
    assert out.ndim == 6 and out.size(0) == 2, "API mismatch: Expected stacked output with first dim equal 2."

# Test 2: Pass input tensor of type float64.
# The dispatch macro (and the kernel) might not handle double types as expected.
def test_input_dtype():
    module = build_kernel()
    batch_size, channels, D, H, W = 2, 3, 8, 8, 8
    input = torch.randn(batch_size, channels, D, H, W, device="cuda", dtype=torch.float64)
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = False
    with pytest.raises(RuntimeError):
        # Expect the kernel launch or dispatch to raise an error due to unsupported data type.
        run_pool3d(module, input, kernel_size, stride, padding, dilation, return_indices, ceil_mode)

# Test 3: Simulate concurrent invocations to trigger potential __constant__ memory race issues.
def test_concurrent_invocations():
    module = build_kernel()
    batch_size, channels, D, H, W = 2, 3, 16, 16, 16
    input1 = torch.randn(batch_size, channels, D, H, W, device="cuda", dtype=torch.float32)
    input2 = torch.randn(batch_size, channels, D, H, W, device="cuda", dtype=torch.float32)
    
    # Different parameters for the two calls.
    params1 = (3, 2, 1, 1, False, False)
    params2 = (3, 1, 0, 2, False, False)
    
    results = [None, None]
    def worker(idx, inp, params):
        results[idx] = run_pool3d(module, inp, *params)
        
    t1 = threading.Thread(target=worker, args=(0, input1, params1))
    t2 = threading.Thread(target=worker, args=(1, input2, params2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # Due to constant memory race, one or both results might be incorrect.
    # Here we perform a simple sanity check: outputs should not be identical.
    # (They are computed on different inputs and with different parameters.)
    assert not torch.allclose(results[0], results[1]), "Concurrent invocations might have caused constant memory data race leading to similar outputs."

# Test 4: Try to use non-uniform (non-scalar) pooling parameters.
# The kernel is written for single int values; passing tuple parameters should raise an error.
def test_nonuniform_parameters():
    module = build_kernel()
    batch_size, channels, D, H, W = 2, 3, 10, 10, 10
    input = torch.randn(batch_size, channels, D, H, W, device="cuda", dtype=torch.float32)
    # Pass a tuple for kernel_size instead of an int.
    kernel_size = (3, 3, 3)
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = False
    with pytest.raises(TypeError):
        run_pool3d(module, input, kernel_size, stride, padding, dilation, return_indices, ceil_mode)

# Test 5: Absence of CUDA error checking.
# Trigger a kernel launch error by providing an input tensor with an unexpected shape.
def test_kernel_launch_error_check():
    module = build_kernel()
    # Provide a 4D tensor instead of 5D.
    input = torch.randn(2, 3, 16, 16, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = False
    with pytest.raises(RuntimeError):
        run_pool3d(module, input, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        
# Note: Issue 2 (missing <limits> include) is a compile‐time error.
# It is expected to be revealed during the build step (build_kernel) rather than at runtime.
# Therefore, we do not write a runtime pytest for it.
