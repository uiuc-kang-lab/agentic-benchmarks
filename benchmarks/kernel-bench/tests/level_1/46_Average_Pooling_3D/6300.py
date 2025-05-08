
import pytest
import torch
import threading
from torch.utils.cpp_extension import load

# Helper function that builds the CUDA kernel module.
def build_kernel():
    cuda_module = load(
        name="pooling_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Input data type support.
# This test passes a double precision tensor to the kernel.
# Since the kernel only supports float, the computation will likely be wrong.
def test_input_type_not_float32():
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size, channels, depth, height, width = 2, 3, 8, 8, 8
    # Create a double tensor (float64) on CUDA.
    input_tensor = torch.randn(batch_size, channels, depth, height, width, device='cuda', dtype=torch.float64)
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should either fail or produce an error because data_ptr<float>() is used.
        cuda_module.forward(input_tensor, kernel_size, stride, padding)
        
# Issue 2: Global constant memory causing race conditions.
# This test launches two concurrent threads, each with different pooling parameters.
# Due to sharing of the global constant memory, the output may be incorrect in one or both calls.
def test_concurrent_kernel_invocations():
    kernel_size1, stride1, padding1 = 3, 2, 1
    kernel_size2, stride2, padding2 = 2, 1, 0
    batch_size, channels, depth, height, width = 2, 3, 16, 16, 16
    input_tensor = torch.randn(batch_size, channels, depth, height, width, device='cuda', dtype=torch.float32)
    cuda_module = build_kernel()
    
    outputs = [None, None]
    
    def run_kernel(idx, k, s, p):
        outputs[idx] = cuda_module.forward(input_tensor, k, s, p)
        torch.cuda.synchronize()
    
    t1 = threading.Thread(target=run_kernel, args=(0, kernel_size1, stride1, padding1))
    t2 = threading.Thread(target=run_kernel, args=(1, kernel_size2, stride2, padding2))
    
    t1.start(); t2.start()
    t1.join(); t2.join()
    
    # Because of the race in updating the constant memory, one or both outputs may not match a locally computed expected result.
    # Here we simply check that the two outputs are different (they should be, if the parameters are different)
    if torch.allclose(outputs[0], outputs[1]):
        pytest.fail("Concurrent invocations produced identical outputs, indicating potential constant memory race issues.")

# Issue 3: Hard-coded divisor.
# This test compares the kernel result with a manually computed result in a situation where border windows are involved.
# For inputs that are not fully covered by the kernel window (i.e. border windows), using a fixed kernel volume may not match
# an implementation that would optionally compute the divisor from the number of valid elements.
def test_divisor_behavior_at_borders():
    kernel_size = 3
    stride = 2
    padding = 1
    # Setup an input tensor filled with ones.
    batch_size, channels, depth, height, width = 1, 1, 5, 5, 5
    input_tensor = torch.ones(batch_size, channels, depth, height, width, device='cuda', dtype=torch.float32)
    cuda_module = build_kernel()
    output = cuda_module.forward(input_tensor, kernel_size, stride, padding)
    torch.cuda.synchronize()
    
    # Manually compute expected output.
    # For a window fully inside the padded area, the sum=1*volume and average=1.
    # At the borders, padded (0) values are assumed so the sum becomes (# valid ones) and average = (# valid ones)/kernel_volume.
    # We mimic the kernel's behavior.
    out_d = (depth + 2 * padding - kernel_size) // stride + 1
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1
    
    expected = torch.empty((batch_size, channels, out_d, out_h, out_w), device='cuda', dtype=torch.float32)
    pool_volume = kernel_size ** 3
    for n in range(batch_size):
        for c in range(channels):
            for d_out in range(out_d):
                for h_out in range(out_h):
                    for w_out in range(out_w):
                        d_start = d_out * stride - padding
                        h_start = h_out * stride - padding
                        w_start = w_out * stride - padding
                        sum_val = 0.0
                        for kd in range(kernel_size):
                            for kh in range(kernel_size):
                                for kw in range(kernel_size):
                                    d_idx = d_start + kd
                                    h_idx = h_start + kh
                                    w_idx = w_start + kw
                                    if 0 <= d_idx < depth and 0 <= h_idx < height and 0 <= w_idx < width:
                                        sum_val += 1.0
                        expected[n, c, d_out, h_out, w_out] = sum_val / pool_volume
    # The kernel divides by pool_volume. In a more general implementation allowing count_include_pad=False,
    # the expected average would be sum_val / (number_of_valid_elements)
    if not torch.allclose(output, expected, atol=1e-5):
        pytest.fail("Kernel average computation (fixed divisor) does not match expected behavior at borders.")

# Issue 4: Missing cudaMemcpyToSymbol error checking.
# We simulate a failure by monkeypatching cudaMemcpyToSymbol.
def test_cudaMemcpyToSymbol_error(monkeypatch):
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size, channels, depth, height, width = 1, 1, 5, 5, 5
    input_tensor = torch.randn(batch_size, channels, depth, height, width, device='cuda', dtype=torch.float32)
    cuda_module = build_kernel()

    # Define a fake cudaMemcpyToSymbol that always returns an error code.
    def fake_cudaMemcpyToSymbol(symbol, src, size):
        # simulate error code not equal to cudaSuccess
        return 1  # non-zero error code
    # Monkey-patch the cudaMemcpyToSymbol in the CUDA extension module.
    # Note: This simulation may require that the kernel module exposes an interface to override or a wrapper.
    # In this test, we assume that we can set an attribute "cudaMemcpyToSymbol" for simulation purposes.
    if hasattr(cuda_module, "cudaMemcpyToSymbol"):
        original = cuda_module.cudaMemcpyToSymbol
        monkeypatch.setattr(cuda_module, "cudaMemcpyToSymbol", fake_cudaMemcpyToSymbol)
        with pytest.raises(RuntimeError):
            cuda_module.forward(input_tensor, kernel_size, stride, padding)
        # Restore original function
        monkeypatch.setattr(cuda_module, "cudaMemcpyToSymbol", original)
    else:
        # If we cannot monkey-patch, then we simply pass the test.
        pytest.skip("CUDA module does not expose cudaMemcpyToSymbol for monkey-patching; cannot test error checking.")

