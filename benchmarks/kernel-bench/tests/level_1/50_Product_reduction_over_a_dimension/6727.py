
import torch
import pytest
import threading
from torch.utils.cpp_extension import load

# Helper to build the CUDA extension module
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Incorrect Indexing when the reduction dimension is not the innermost dimension.
def test_incorrect_indexing():
    my_module = build_kernel()
    # Create a tensor that is contiguous and non-trivial. 
    # Use shape (4, 10, 6) and reduce over dimension 1 (which is not the last dimension).
    x = torch.randn(4, 10, 6, device='cuda', dtype=torch.float32)
    # Expected result using torch.prod.
    expected = torch.prod(x, dim=1)
    # Call kernel forward.
    result = my_module.forward(x, 1)
    torch.cuda.synchronize()
    # Because of incorrect indexing, result will likely differ.
    assert not torch.allclose(result, expected, atol=1e-5), \
        "Test expected an indexing issue to cause an incorrect product reduction!"

# Test 2: Concurrent kernel launches causing race conditions with constant memory.
def test_concurrent_constant_memory_race_condition():
    my_module = build_kernel()
    # Using two different input sizes and reduction dimensions concurrently.
    x1 = torch.randn(8, 7, device='cuda', dtype=torch.float32)
    x2 = torch.randn(8, 9, device='cuda', dtype=torch.float32)

    # Expected outputs
    expected1 = torch.prod(x1, dim=1)
    expected2 = torch.prod(x2, dim=1)

    results = [None, None]
    def run_kernel(idx, x, dim):
        results[idx] = my_module.forward(x, dim)
        torch.cuda.synchronize()

    # Launch in concurrent threads.
    t1 = threading.Thread(target=run_kernel, args=(0, x1, 1))
    t2 = threading.Thread(target=run_kernel, args=(1, x2, 1))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Due to constant memory race conditions, at least one result might be incorrect.
    cond1 = torch.allclose(results[0], expected1, atol=1e-5)
    cond2 = torch.allclose(results[1], expected2, atol=1e-5)
    assert not (cond1 and cond2), "Expected concurrent kernel launches to expose constant memory race condition!"

# Test 3: Kernel Launch Error Checking by providing unsupported data type.
def test_unsupported_dtype():
    my_module = build_kernel()
    # Create a tensor of type double.
    x = torch.randn(16, 10, device='cuda', dtype=torch.float64)
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x, 1)
    # The CHECK_INPUT macro does not check type, but the kernel expects float.
    # This should ultimately cause a problem (e.g. missized element loads or assertion in CUDA).

# Test 4: Non-Contiguous Input Tensor
def test_non_contiguous_input():
    my_module = build_kernel()
    x = torch.randn(8, 9, device='cuda', dtype=torch.float32)
    # Make x non-contiguous by a transpose.
    x_t = x.t()
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x_t, 0)
    # The CHECK_INPUT macro should trigger error for non-contiguous input.

if __name__ == '__main__':
    pytest.main([__file__])
