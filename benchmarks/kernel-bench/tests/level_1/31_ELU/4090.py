
import torch
import pytest
import threading
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="elu_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Misaligned memory access
def test_misaligned_input():
    my_module = build_kernel()
    # Create a tensor that is likely to be aligned
    base = torch.randn(1025, dtype=torch.float32, device='cuda')
    # Create a misaligned view by slicing off the first element
    x = base.narrow(0, 1, 1024).contiguous()
    alpha = 1.0
    # Forward through our kernel & compute expected result via PyTorch
    out = my_module.forward(x, alpha)
    out_ref = torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    # The misaligned load may yield wrong results (or even silently corrupt memory)
    # so we expect a difference between the two.
    assert not torch.allclose(out, out_ref, atol=1e-4), \
        "Kernel unexpectedly produced correct result on misaligned tensor. Issue 1 is not triggered."

# Test 2: Race condition with concurrent different alpha values
def test_race_condition_constant_alpha():
    my_module = build_kernel()
    # Prepare a tensor
    x = torch.randn(4096, dtype=torch.float32, device='cuda')
    # Expected outputs for two different alpha values computed on CPU
    def compute_expected(tensor, a):
        return torch.where(tensor > 0, tensor, a * (torch.exp(tensor) - 1))
    
    results = {}

    def worker(name, alpha_value):
        # Use a separate stream for concurrent execution
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            res = my_module.forward(x, alpha_value)
        torch.cuda.synchronize()
        results[name] = res

    # Launch two threads concurrently with different alpha values
    t1 = threading.Thread(target=worker, args=("alpha1", 1.0))
    t2 = threading.Thread(target=worker, args=("alpha2", 2.0))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Compare each result with its expected outcome.
    out1_ref = compute_expected(x, 1.0)
    out2_ref = compute_expected(x, 2.0)
    
    # Because of the race condition the outputs may be wrong.
    err1 = (results["alpha1"] - out1_ref).abs().max().item()
    err2 = (results["alpha2"] - out2_ref).abs().max().item()
    # We assume that if race condition occurs the errors will be significant.
    assert err1 > 1e-4 or err2 > 1e-4, \
        "Kernel produced correct results for concurrent alpha values. Issue 2 is not triggered."

# Test 3: Input tensor with unsupported data type (double)
def test_input_tensor_dtype():
    my_module = build_kernel()
    # Create a double tensor. The kernel expects float but does not check type.
    x = torch.randn(1024, dtype=torch.float64, device='cuda')
    alpha = 1.0
    with pytest.raises(RuntimeError):
        # This should raise an error or produce a wrong result due to misinterpretation of data.
        out = my_module.forward(x, alpha)
        torch.cuda.synchronize()

# Test 4: No kernel launch error checked (simulate an error by sending an empty tensor)
def test_empty_input_tensor():
    my_module = build_kernel()
    # Passing an empty tensor should normally work, but if there is a kernel launch issue,
    # the error might be hidden. Here we simply run with an empty tensor.
    x = torch.empty((0,), dtype=torch.float32, device='cuda')
    alpha = 1.0
    out = my_module.forward(x, alpha)
    # Expect an empty tensor output
    assert out.numel() == 0, "Kernel output should be an empty tensor but is not."

if __name__ == "__main__":
    pytest.main([__file__])
