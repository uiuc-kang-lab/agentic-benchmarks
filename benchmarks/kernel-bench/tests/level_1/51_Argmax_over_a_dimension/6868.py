
import pytest
import torch
from torch.utils.cpp_extension import load
import threading
import time

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger dtype issue by passing an input tensor with non-float32 type.
def test_dtype_issue():
    my_module = build_kernel()
    # Create a tensor with dtype double, which is not supported by the kernel.
    x = torch.randn(16, 256, 256, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError, match="Only float32 is supported"):
        # This should throw an exception since our kernel only supports float32.
        my_module.forward(x, 1)
        
# Test 2: Trigger constant memory concurrency issue.
# We launch two kernel invocations concurrently with different 'dim' parameters.
# Due to using __constant__ memory, if the settings interfere, one or both results may be wrong.
def test_concurrent_constant_memory_issue():
    my_module = build_kernel()
    
    # Function to run the kernel with a given dimension and store result.
    result_holder = {}
    def run_kernel(dim, key):
        # Create a random tensor
        x = torch.randn(4, 8, 16, dtype=torch.float32, device="cuda")
        # Let the model use the given dim for argmax.
        res = my_module.forward(x, dim)
        # Force synchronization and store result.
        torch.cuda.synchronize()
        result_holder[key] = res.clone()
    
    # Launch two threads concurrently with different 'dim' parameters.
    thread1 = threading.Thread(target=run_kernel, args=(1, "first"))
    thread2 = threading.Thread(target=run_kernel, args=(2, "second"))
    
    thread1.start()
    # Introduce a slight delay to increase the chance of concurrent execution.
    time.sleep(0.01)
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    # Check that the results have the expected shapes:
    # For dim=1, the output should have shape [4,16]
    # For dim=2, the output should have shape [4,8]
    res_first = result_holder.get("first")
    res_second = result_holder.get("second")
    
    assert res_first is not None and list(res_first.shape) == [4, 16], (
        f"Expected output shape [4,16] for dim=1, got {list(res_first.shape)}"
    )
    assert res_second is not None and list(res_second.shape) == [4, 8], (
        f"Expected output shape [4,8] for dim=2, got {list(res_second.shape)}"
    )
    
    # In a properly designed kernel that passes dimension parameters via kernel arguments,
    # the outputs would be independent. Here, if constant memory is overwritten concurrently,
    # one expected outcome is that one of these outputs might be incorrect.
    # (This test may sporadically fail on systems with heavy concurrency issues.)
    
# Test 3: Trigger issue with an empty dimension.
def test_empty_dim_issue():
    my_module = build_kernel()
    # Create an input tensor where the chosen dim (here, dim 1) is empty.
    x = torch.randn(10, 0, 5, dtype=torch.float32, device="cuda")
    # When the reduction dimension is empty, a proper implementation should raise an error.
    # Our current kernel does not check for this and will return an output (likely filled with zeros or garbage)
    # Here we check if the output has the expected size (removing the empty dimension) but also warn on meaningless result.
    res = my_module.forward(x, 1)
    torch.cuda.synchronize()
    # Expected shape should be [10,5] because dim 1 is removed.
    assert list(res.shape) == [10,5], f"Expected output shape [10,5] when reducing empty dim, got {list(res.shape)}"
    # Additionally, check that the output values are not valid since there was no element to reduce.
    # We expect the kernel might return 0s (or garbage) but it should be considered an error in a robust implementation.
    if torch.any(res != 0):
        pytest.fail("Kernel did not handle empty reduction dimension correctly; expected all zeros or an error.")

