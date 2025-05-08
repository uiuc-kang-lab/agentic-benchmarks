
import pytest
import torch
from torch.utils.cpp_extension import load
import math

# Build the CUDA extension from the kernel.cu file.
def build_kernel():
    cuda_module = load(
        name="gelu_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference GELU function as defined in the PyTorch code.
def gelu_ref(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# Issue 1: Kernel only supports float32.
def test_dtype_mismatch():
    my_module = build_kernel()
    # Create a double tensor (float64), which is not supported by the kernel.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # Compute reference using torch's GELU (on double).
    y_ref = gelu_ref(x)
    # Running the kernel on a double tensor leads to reinterpretation of data as float.
    y_cuda = my_module.forward(x)
    torch.cuda.synchronize()
    # Since the kernel misinterprets double data as float, the result will not match.
    # We expect the maximum difference to be large.
    assert not torch.allclose(y_cuda.to(torch.float64), y_ref, atol=1e-3), (
        "Kernel should fail or produce significantly different results when using double dtype."
    )

# Issue 2: Assumes 16-byte alignment (vectorized loads/stores with float4).
def test_misaligned_memory():
    my_module = build_kernel()
    # Create a larger tensor and then take a narrow slice to force a nonzero storage offset.
    # This can often cause the returned tensor's data pointer to be misaligned relative to 16 bytes.
    x_full = torch.randn(1025, device="cuda", dtype=torch.float32)
    # Slice off the first element so that the pointer is offset by 4 bytes (size of one float).
    x = x_full.narrow(0, 1, 1024).contiguous()
    # Verify misalignment: data pointer modulo 16 should not be 0.
    ptr = x.data_ptr()
    assert ptr % 16 != 0, "Test setup failure: the tensor is unexpectedly 16-byte aligned."
    
    y_ref = gelu_ref(x)
    y_cuda = my_module.forward(x)
    torch.cuda.synchronize()
    # Due to potential misaligned memory access, the kernel may produce incorrect results.
    # Thus, we expect the output to differ from the reference.
    diff = (y_cuda - y_ref).abs().max().item()
    assert diff > 1e-3, (
        f"Kernel output appears correct despite misaligned memory (max diff {diff}). "
        "This test was meant to uncover misaligned access issues."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
