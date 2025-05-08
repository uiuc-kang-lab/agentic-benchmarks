
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports float32. Testing with double input.
def test_kernel_wrong_dtype():
    module = build_kernel()
    # Create double precision inputs on CUDA.
    A = torch.randn(128, 64, dtype=torch.float64, device="cuda")
    B = torch.randn(64, 128, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError):
        # Expect a runtime error due to type mismatch since kernel only supports float
        C = module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Kernel launch error checking is missing.
# We'll try to trigger an out-of-bound memory access by providing tensors that are non-contiguous
def test_noncontiguous_input():
    module = build_kernel()
    # Create contiguous tensors, then take a non-contiguous slice.
    A_full = torch.randn(128, 64, dtype=torch.float32, device="cuda")
    B_full = torch.randn(64, 128, dtype=torch.float32, device="cuda")
    A = A_full[:, ::2]  # slicing makes the tensor non-contiguous
    B = B_full[:, ::2]  # slicing makes the tensor non-contiguous
    # The kernel does not check for contiguity;
    # if the computed strides do not match expected contiguous access, the result may be wrong.
    C = module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result using torch.matmul. These will likely not match due to wrong stride usage.
    C_ref = torch.matmul(A, B)
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel should produce invalid results for non-contiguous inputs."

# Issue 3: Lack of explicit kernel launch error checking.
# We simulate this by intentionally providing matrices with dimensions that force the kernel
# to use an output tensor whose leading dimension (ldc) may not match the expected layout.
def test_invalid_memory_layout():
    module = build_kernel()
    # Create matrices that are non-contiguous in a way that their strides do not correspond to a standard dense matrix.
    A_base = torch.randn(256, 256, dtype=torch.float32, device="cuda")
    B_base = torch.randn(256, 256, dtype=torch.float32, device="cuda")
    # Transpose the tensor to break contiguity (without calling .contiguous()).
    A_t = A_base.t()  # non-contiguous view
    B_t = B_base.t()  # non-contiguous view
    # The kernel uses the stride from the tensor, which might work for transpose,
    # but our kernel assumptions may fail in more complex settings.
    # We compute the results and compare with torch.matmul.
    C = module.forward(A_t, B_t)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A_t, B_t)
    # We expect a difference due to misinterpretation of strides by the kernel.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel should produce invalid results for non-standard (transposed non-contiguous) inputs."

if __name__ == "__main__":
    pytest.main([__file__])
