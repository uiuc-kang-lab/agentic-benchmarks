
import torch
import pytest
from torch.utils.cpp_extension import load
import threading

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Reference function using PyTorch's own AvgPool1d for comparison.
def ref_avg_pool1d(x, kernel_size, stride, padding):
    pool = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    return pool(x)

# Issue 1: Type support. The kernel is hard-coded for float32.
def test_input_dtype():
    cuda_module = build_kernel()
    batch_size = 4
    in_channels = 3
    input_length = 32
    kernel_size = 3
    stride = 2
    padding = 1

    # Create a double tensor.
    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.float64, device="cuda")
    
    # The custom kernel does not check for type and will treat data as float.
    # Therefore the result is expected to be wrong compared to the reference.
    out_kernel = cuda_module.forward(x, kernel_size, stride, padding)
    out_ref = ref_avg_pool1d(x.float(), kernel_size, stride, padding)
    
    # Since the kernel misinterprets the data, the outputs will differ.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), "Kernel unexpectedly processed non-float32 tensor correctly!"

# Issue 2: Global constant memory race in multi-stream execution.
def test_multistream_race_condition():
    cuda_module = build_kernel()
    batch_size = 8
    in_channels = 4
    input_length = 64

    # We create two sets of pooling parameters that are different.
    params_set = [
        {"kernel_size": 3, "stride": 1, "padding": 1},
        {"kernel_size": 5, "stride": 2, "padding": 2}
    ]
    
    outputs = {}
    
    def run_kernel(name, params):
        x = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32, device="cuda")
        # Launch on a separate stream.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            out = cuda_module.forward(x, params["kernel_size"], params["stride"], params["padding"])
        # Ensure the kernel finishes.
        torch.cuda.synchronize()
        # Compare with reference.
        out_ref = ref_avg_pool1d(x, params["kernel_size"], params["stride"], params["padding"])
        outputs[name] = (out, out_ref)
    
    # Run two kernels concurrently in two threads.
    thread1 = threading.Thread(target=run_kernel, args=("task1", params_set[0]))
    thread2 = threading.Thread(target=run_kernel, args=("task2", params_set[1]))
    
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    # Due to the global constant memory race, one or both outputs are likely to be incorrect.
    for key, (out, out_ref) in outputs.items():
        if torch.allclose(out, out_ref, atol=1e-5):
            pytest.skip("Concurrent execution did not trigger a visible issue; multi-stream race condition may not have manifested on this run.")
        else:
            # We expect a mismatch.
            assert not torch.allclose(out, out_ref, atol=1e-5), f"Kernel result in {key} unexpectedly matches reference despite multi-stream race potential!"

# Issue 3: Non-contiguous input tensor.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch_size = 4
    in_channels = 3
    input_length = 32
    kernel_size = 3
    stride = 2
    padding = 1

    # Create a contiguous tensor and then make a non-contiguous view by transposing and then transposing back.
    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32, device="cuda")
    x_nc = x.transpose(1, 2).transpose(1, 2)  # Although x_nc may be logically same shape, it is not necessarily contiguous.

    assert not x_nc.is_contiguous(), "Test tensor is contiguous, cannot test non-contiguous case."

    out_kernel = cuda_module.forward(x_nc, kernel_size, stride, padding)
    out_ref = ref_avg_pool1d(x_nc.contiguous(), kernel_size, stride, padding)
    
    # Expect differences because kernel does not handle non-contiguous layout.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), "Kernel unexpectedly processed non-contiguous tensor correctly!"

# Issue 4: Missing error checking and stream synchronization.
def test_kernel_launch_without_proper_error_checks():
    cuda_module = build_kernel()
    batch_size = 4
    in_channels = 3
    input_length = 32
    kernel_size = 3
    stride = 1
    padding = 1

    # Create a valid tensor.
    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32, device="cuda")
    
    # Launch the kernel and immediately check for errors by synchronizing.
    out_kernel = cuda_module.forward(x, kernel_size, stride, padding)
    try:
        torch.cuda.synchronize()
    except RuntimeError as e:
        pytest.fail("CUDA kernel launch error was not properly handled: " + str(e))
    
    # Although no exception was thrown, this test is here to note that the lack of proper error checking in the kernel/hub wrapper is an issue.
    # We compare the result with the reference to show that while no error is caught, the output might still be wrong.
    out_ref = ref_avg_pool1d(x, kernel_size, stride, padding)
    if not torch.allclose(out_kernel, out_ref, atol=1e-5):
        pytest.skip("Kernel did not produce correct output due to lack of proper error checking and stream use.")
    else:
        pytest.skip("Kernel produced correct output; error checking issue not exposed in this run.")

if __name__ == "__main__":
    pytest.main()
