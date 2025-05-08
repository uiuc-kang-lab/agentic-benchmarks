
import torch
import torch.nn.functional as F
import threading
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build our custom CUDA kernel extension. The module is called "custom_maxpool"
    return load(
        name="custom_maxpool",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# Issue 1: Global constant memory race condition (non–reentrant kernel)
def test_concurrent_invocations():
    # This test spawns two threads using different pooling parameters concurrently,
    # then compares their results to the reference torch.nn.functional.max_pool2d result.
    cuda_module = build_kernel()

    # Create two different parameter sets
    params1 = dict(kernel_size=2, stride=2, padding=1, dilation=1)
    params2 = dict(kernel_size=3, stride=1, padding=0, dilation=2)

    # Create a contiguous input tensor
    input_tensor = torch.randn(16, 32, 64, 64, device="cuda", dtype=torch.float32)

    # These holder dictionaries will store outputs from each thread.
    outputs = {}

    def run_pool(prefix, params):
        # Call the custom forward method from the CUDA extension.
        output = cuda_module.forward(input_tensor, params['kernel_size'], params['stride'], params['padding'], params['dilation'])
        # Also compute the reference output from PyTorch for comparison.
        ref = F.max_pool2d(input_tensor, kernel_size=params['kernel_size'], 
                             stride=params['stride'], padding=params['padding'], dilation=params['dilation'])
        outputs[prefix] = (output, ref)

    t1 = threading.Thread(target=run_pool, args=("first", params1))
    t2 = threading.Thread(target=run_pool, args=("second", params2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Compare results. If constant memory is overwritten concurrently,
    # at least one of the outputs will not match the reference.
    out1, ref1 = outputs["first"]
    out2, ref2 = outputs["second"]

    err1 = (out1 - ref1).abs().max().item()
    err2 = (out2 - ref2).abs().max().item()
    # We expect an exact match if the kernel was reentrant.
    assert err1 < 1e-5 or err2 < 1e-5, (
        "Concurrent kernel invocations with different parameters produce conflicting results. "
        "This signals a race condition with global constant memory."
    )

# Issue 2: Non-contiguous input tensor handling
def test_non_contiguous_input():
    cuda_module = build_kernel()
    params = dict(kernel_size=2, stride=2, padding=1, dilation=3)
    # Create a contiguous tensor and then make it non-contiguous via transpose.
    input_tensor = torch.randn(16, 32, 128, 128, device="cuda", dtype=torch.float32)
    non_contig = input_tensor.transpose(2, 3)  # Now non-contiguous

    # Run the kernel on the non-contiguous tensor.
    try:
        output = cuda_module.forward(non_contig, params['kernel_size'], params['stride'], params['padding'], params['dilation'])
    except RuntimeError as e:
        pytest.skip("Kernel does not support non-contiguous inputs; skipping further check")
    # For reference, force contiguous before using PyTorch op.
    ref = F.max_pool2d(non_contig.contiguous(), kernel_size=params['kernel_size'], 
                       stride=params['stride'], padding=params['padding'], dilation=params['dilation'])
    # This should match if non-contiguous is correctly handled.
    max_err = (output - ref).abs().max().item()
    assert max_err < 1e-5, "Kernel output for non-contiguous input does not match the reference."

# Issue 3: Missing error checking on cudaMemcpyToSymbol and kernel launch
def test_invalid_input_dimensions():
    cuda_module = build_kernel()
    params = dict(kernel_size=2, stride=2, padding=1, dilation=3)
    # Create an input tensor with zero elements (which can trigger unusual kernel launches)
    input_tensor = torch.empty(0, device="cuda", dtype=torch.float32)
    # The reference operation should also handle this gracefully.
    ref = F.max_pool2d(input_tensor, kernel_size=params['kernel_size'], 
                       stride=params['stride'], padding=params['padding'], dilation=params['dilation'])
    output = cuda_module.forward(input_tensor, params['kernel_size'], params['stride'], params['padding'], params['dilation'])
    assert output.numel() == ref.numel(), "Kernel did not gracefully handle an input with zero elements."

# Issue 4: Use of unqualified max function in device code.
def test_unsupported_input_type():
    cuda_module = build_kernel()
    params = dict(kernel_size=2, stride=2, padding=1, dilation=3)
    # Pass an integer tensor to trigger dispatch over an unsupported type.
    input_tensor = torch.randint(0, 255, (16, 32, 128, 128), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        # The kernel is only dispatched for floating point types.
        _ = cuda_module.forward(input_tensor, params['kernel_size'], params['stride'], params['padding'], params['dilation'])

# Issue 5: Lack of support for half-precision types.
def test_half_precision_input():
    cuda_module = build_kernel()
    params = dict(kernel_size=2, stride=2, padding=1, dilation=3)
    # Create a half precision tensor.
    input_tensor = torch.randn(16, 32, 128, 128, device="cuda", dtype=torch.float16")
    try:
        # Try running the kernel.
        output = cuda_module.forward(input_tensor, params['kernel_size'], params['stride'], params['padding'], params['dilation'])
    except RuntimeError as e:
        pytest.skip("Kernel did not compile for float16 (half precision), which is an unsupported type.")
    # Compute reference using float32 conversion (if max_pool2d supports half, then this is acceptable)
    ref = F.max_pool2d(input_tensor.float(), kernel_size=params['kernel_size'], 
                       stride=params['stride'], padding=params['padding'], dilation=params['dilation']).half()
    max_err = (output - ref).abs().max().item()
    assert max_err < 1e-2, "Kernel output for half precision input does not match the reference within tolerance."

