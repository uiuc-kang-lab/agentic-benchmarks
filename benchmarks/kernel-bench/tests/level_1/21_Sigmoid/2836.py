
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

def test_misaligned_memory(kernel_module):
    # Issue 1: Misaligned memory: Create a tensor view with an offset to force misalignment.
    # Allocate a tensor with one extra element and slice it from index 1.
    base = torch.randn(16385, device="cuda", dtype=torch.float32)
    # Using slice to force misalignment: the underlying pointer might not be 128-bit aligned.
    x = base[1:]
    # Some tensors created this way are still contiguous, but their data_ptr offset causes misalignment for vector loads.
    out = kernel_module.forward(x)
    expected = torch.sigmoid(x)
    # We expect differences or potential errors because the kernel assumed proper alignment.
    # Here we check that the results do not exactly match.
    assert not torch.allclose(out, expected, atol=1e-5), (
        "Expected discrepancy due to misaligned memory, but the results are too close."
    )

def test_non_contiguous_tensor(kernel_module):
    # Issue 4: Noncontiguous tensor: The kernel assumes contiguous memory.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    # Create a noncontiguous tensor by transposing.
    x_noncontig = x.t()  
    assert not x_noncontig.is_contiguous(), "Tensor should be noncontiguous"
    out = kernel_module.forward(x_noncontig)
    expected = torch.sigmoid(x_noncontig)
    # A noncontiguous tensor may lead to wrong results (or even subtle memory errors)
    assert not torch.allclose(out, expected, atol=1e-5), (
        "Noncontiguous tensor did not trigger the expected difference in output."
    )

def test_unsupported_dtype(kernel_module):
    # Issue 2: Unsupported data type: Using torch.half should trigger an error.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.half)
    with pytest.raises(RuntimeError):
        # The kernel only dispatches for float32 and float64; half should cause a failure.
        _ = kernel_module.forward(x)

def test_strict_aliasing_union(kernel_module):
    # Issue 3: The union-based vectorized loads/stores might lead to aliasing issues.
    # While such issues are often optimized away in CUDA code across simplified tests,
    # We try to stress the kernel with a large tensor to see potential undefined behavior.
    x = torch.randn(1 << 16, device="cuda", dtype=torch.float32)
    out = kernel_module.forward(x)
    expected = torch.sigmoid(x)
    # If strict aliasing creates any subtle corruption, results will deviate.
    assert not torch.allclose(out, expected, atol=1e-5), (
        "Union-based vectorized access did not trigger an expected error in output."
    )

def test_no_kernel_error_checking(kernel_module):
    # Issue 5: No error checking / synchronization can hide kernel launch errors.
    # We purposefully launch on a huge tensor that may stress resource limits.
    try:
        x = torch.randn(1 << 28, device="cuda", dtype=torch.float32)
        out = kernel_module.forward(x)
        expected = torch.sigmoid(x)
        # Even if no exception is thrown, we might get incorrect results.
        assert not torch.allclose(out, expected, atol=1e-5), (
            "Huge tensor did not trigger an expected error or discrepancy in results."
        )
    except Exception as e:
        # If an error is raised, the lack of error checking in the kernel is highlighted.
        pytest.skip("Kernel launch failed as expected due to resource limits: " + str(e))
