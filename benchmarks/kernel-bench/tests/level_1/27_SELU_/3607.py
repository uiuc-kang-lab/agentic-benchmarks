
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="selu_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger an error by using a half precision input.
# Since the kernel does not support half (float16) inputs, we expect a RuntimeError.
def test_half_precision_not_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = build_kernel()
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # The kernel dispatch will fail because float16 is not covered.
        _ = mod.forward(x)
        
# Test case 2: Check for lack of kernel launch error checking.
# In this test we provide an input tensor that is on the CPU.
# The host function is expected to check for a CUDA tensor (via input.is_cuda()).
def test_input_tensor_cpu_error():
    mod = build_kernel()
    x = torch.randn(1024, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _ = mod.forward(x)

# Test case 3: Provide a non-contiguous input tensor.
# Since the kernel assumes contiguity, a non-contiguous input may result in incorrect computation.
def test_non_contiguous_tensor():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mod = build_kernel()
    # Create a contiguous tensor and then create a non-contiguous view by transposing.
    x = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    non_contig = x.t()  # non-contiguous view
    # Compute reference using PyTorch SELU on a contiguous copy.
    ref = torch.selu(non_contig.contiguous()).to(non_contig.device)
    # Run the CUDA kernel on the non-contiguous tensor.
    out = mod.forward(non_contig)
    # If the kernel assumed contiguity, the result will be different.
    # We use a loose tolerance to capture differences.
    if torch.allclose(out, ref, atol=1e-5):
        pytest.fail("Kernel unexpectedly produced correct results for a non-contiguous input."
                    " This suggests the kernel may be implicitly assuming contiguity."
                    " In a general setting the kernel should either support non-contiguous tensors or check for contiguity.")

