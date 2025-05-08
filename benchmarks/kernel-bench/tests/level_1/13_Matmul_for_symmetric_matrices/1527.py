
import pytest
import torch
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

# Issue 1: Test that using a non-float32 tensor triggers an error.
def test_dtype_check():
    N = 256
    # Create double precision symmetric matrices.
    A = torch.randn(N, N, dtype=torch.float64, device='cuda')
    A = (A + A.t()) / 2
    B = torch.randn(N, N, dtype=torch.float64, device='cuda')
    B = (B + B.t()) / 2

    my_module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        # This call is expected to throw because the kernel uses data_ptr<float>(),
        # causing a misinterpretation of the underlying data.
        my_module.forward(A, B)
    assert "expected" in str(excinfo.value).lower() or "float" in str(excinfo.value).lower(), \
        "The kernel did not complain about non-float32 data types as expected."

# Issue 2: Test that providing non-square matrices triggers an error.
def test_square_matrix_check():
    # Create non-square matrices.
    A = torch.randn(256, 128, dtype=torch.float32, device='cuda')
    B = torch.randn(128, 256, dtype=torch.float32, device='cuda')

    my_module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        # The forward function should raise an error because of the TORCH_CHECK that enforces A and B to be square.
        my_module.forward(A, B)
    assert "square" in str(excinfo.value).lower(), "The kernel did not complain about non-square matrices as expected."

# Issue 3: Since there is no explicit error check after the CUDA kernel launch,
# we attempt to induce an error by providing inputs with a size that will cause a kernel launch error.
# One common method is to provide an empty tensor.
def test_kernel_launch_error_check():
    N = 0  # Empty matrix dimensions
    A = torch.empty((N, N), dtype=torch.float32, device='cuda')
    B = torch.empty((N, N), dtype=torch.float32, device='cuda')

    my_module = build_kernel()
    # The kernel may not crash when provided empty tensors,
    # but a robust implementation should check and error out properly.
    # Here we check if the output is also empty.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    assert C.numel() == 0, "Kernel did not handle empty input correctly, which might hide launch errors."

if __name__ == '__main__':
    pytest.main([__file__])
