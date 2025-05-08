
import pytest
import torch
from torch.cuda import is_available
from torch.utils.cpp_extension import load
import time
import threading

# Skip tests if CUDA is not available.
pytestmark = pytest.mark.skipif(not is_available(), reason="CUDA not available")

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_incorrect_dtype():
    # Test Issue 1: Incorrect tensor dtype (double instead of float32).
    module = build_kernel()
    N = 1024
    x = torch.randn(N, device='cuda', dtype=torch.double)
    # Expect an error because the kernel assumes float*.
    with pytest.raises(RuntimeError) as excinfo:
        # This should trigger an incorrect memory interpretation or a CHECK failure if implemented.
        module.forward(x, 1.0)
    assert "float" in str(excinfo.value) or "dtype" in str(excinfo.value), "Expected error regarding dtype"

def test_non_contiguous():
    # Test Issue 1 (non-contiguous input triggering CHECK_CONTIGUOUS failure).
    module = build_kernel()
    x = torch.randn(64, 64, device='cuda', dtype=torch.float32)
    x_non_contig = x.t()  # Transpose makes it non-contiguous.
    with pytest.raises(RuntimeError) as excinfo:
        module.forward(x_non_contig, 1.0)
    assert "contiguous" in str(excinfo.value), "Expected error regarding non-contiguous tensor"

def test_concurrent_alpha():
    # Test Issue 3: Concurrent launches with different alpha values.
    # If constant memory is overwritten concurrently, results from one stream could be affected by the other.
    module = build_kernel()
    N = 1024 * 10
    # Give a tensor with negative values so that ELU (exp-1)*alpha takes noticeable effect.
    x1 = -torch.abs(torch.randn(N, device='cuda', dtype=torch.float32))
    x2 = -torch.abs(torch.randn(N, device='cuda', dtype=torch.float32))
    
    # Expected outputs computed with known alpha values.
    def compute_expected(x, alpha):
        # Note: ELU: x if x > 0 else alpha*(exp(x)-1)
        return torch.where(x > 0, x, alpha*(x.exp()-1))
    
    expected1 = compute_expected(x1, 1.0)
    expected2 = compute_expected(x2, 2.0)
    
    out1 = None
    out2 = None

    # Run two kernel calls concurrently in separate threads.
    def run_kernel(result_container, tensor, alpha, delay=0):
        # Introduce an optional delay to force overlap.
        if delay:
            time.sleep(delay)
        result = module.forward(tensor, alpha)
        torch.cuda.synchronize()
        result_container.append(result)
        
    result_list1 = []
    result_list2 = []

    thread1 = threading.Thread(target=run_kernel, args=(result_list1, x1, 1.0, 0.0))
    thread2 = threading.Thread(target=run_kernel, args=(result_list2, x2, 2.0, 0.01))
    
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    
    # Check that the outputs differ accordingly.
    out1 = result_list1[0]
    out2 = result_list2[0]
    
    # Due to potential race condition on constant memory, at least one of the outputs may be computed with the wrong alpha value.
    # We flag the error if the outputs are not close to the expected result.
    err1 = (out1 - expected1).abs().max().item()
    err2 = (out2 - expected2).abs().max().item()
    
    atol = 1e-5
    # One or both comparisons might fail if constant memory is overwritten.
    assert (err1 < atol) and (err2 < atol), (
        f"Concurrent launches produced mismatched results: max error for alpha=1.0={err1}, for alpha=2.0={err2}. "
        "This indicates a potential race condition with constant memory usage."
    )
