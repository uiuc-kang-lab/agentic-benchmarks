
import torch
import pytest
from torch.utils.cpp_extension import load

# Build and load our custom CUDA kernel extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="max_reduce_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# 1. Test when the input tensor is non-contiguous.
def test_non_contiguous_input():
    # Create a contiguous tensor then make it non-contiguous via transpose.
    x = torch.randn(4, 8, 16, device="cuda")
    x_non_contig = x.transpose(0, 1)  # Now non-contiguous.
    # We'll reduce along dimension 1 (which is non-contiguous)
    dim = 1
    kernel = build_kernel()
    # Compute with our CUDA kernel and compare with torch.max.
    try:
        out_kernel = kernel.forward(x_non_contig, dim)
    except Exception as e:
        pytest.skip("Kernel raised an exception on non-contiguous input, which is expected.")
    out_ref = torch.max(x_non_contig, dim=dim)[0]
    # In a correct implementation these should agree.
    # Here we expect a discrepancy because the kernel assumes contiguous input.
    if torch.allclose(out_kernel, out_ref):
        pytest.fail("Kernel produced correct results on non-contiguous input despite assuming contiguous layout.")

# 2. Test half precision input.
def test_half_precision_input():
    # Create a contiguous half precision tensor.
    x = torch.randn(8, 16, 32, device="cuda").half()
    dim = 2
    kernel = build_kernel()
    # It is possible that __ldg on half leads to incorrect results or compile/runtime issues.
    with pytest.raises(Exception):
        # Expecting that the kernel will most likely throw an exception or produce a wrong result.
        _ = kernel.forward(x, dim)

# 3. Test empty reduction dimension.
def test_empty_reduction_dimension():
    # Create a tensor where the reduction dimension has size 0.
    # For example, shape where dimension 1 is zero.
    x = torch.randn(4, 0, 32, device="cuda")
    dim = 1
    kernel = build_kernel()
    with pytest.raises(Exception):
        # Expect the kernel to attempt to read __ldg(input + start_idx) with dim_size==0 and fail.
        _ = kernel.forward(x, dim)

# 4. Test potential ambiguity with max() for non-standard data types.
def test_unsupported_data_type():
    # Using a complex tensor (if available) to see if the kernel fails due to ambiguous max operation.
    # Note: The kernel does not support complex types.
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    try:
        x = torch.randn(4, 16, 32, device="cuda", dtype=torch.complex64)
    except TypeError:
        pytest.skip("Complex tensor creation not supported on this PyTorch version.")
    dim = 2
    kernel = build_kernel()
    with pytest.raises(Exception):
        _ = kernel.forward(x, dim)

# 5. Test lack of kernel launch error checking by triggering an invalid launch.
def test_invalid_dimension_reduction():
    # Pass an invalid dimension (e.g., out of bounds) to see if the kernel fails gracefully.
    x = torch.randn(8, 16, 32, device="cuda")
    invalid_dim = 10  # out of bounds for a 3D tensor
    kernel = build_kernel()
    with pytest.raises(Exception):
        _ = kernel.forward(x, invalid_dim)

# 6. Test potential integer overflow in indexing.
def test_index_overflow():
    # Simulate a scenario with huge outer_size*inner_size.
    # We create a tensor with extremely large outer_size and inner_size but small reduction dim.
    # Note: It is not trivial to create a tensor that leads to a real overflow in practice due to memory limits.
    # Instead, we mimic the condition by monkey-patching input.size to return huge numbers.
    class FakeTensor:
        def __init__(self):
            self._sizes = [2**30, 3, 2**30]  # outer_size=2**30, dim_size=3, inner_size=2**30
            self.device = torch.device("cuda")
            self.dtype = torch.float32
            self.scalar_type = lambda: torch.float32
        def size(self, idx):
            return self._sizes[idx]
        def sizes(self):
            return self._sizes
        @property
        def dim(self):
            return len(self._sizes)
        def options(self):
            return torch.empty(0).options()
        def data_ptr(self):
            # Not used by our test.
            return 0

    # We can't actually run the kernel because we cannot allocate a tensor of that size.
    # So we simulate by constructing an input tensor and then monkey-patching its sizes.
    x = torch.randn(2, 3, 2, device="cuda")
    # Monkey-patch sizes to simulate huge outer_size and inner_size.
    x_fake = x.clone()
    x_fake.get_device = lambda: 0
    x_fake.size = lambda idx: {0: 2**30, 1: 3, 2: 2**30}[idx]
    x_fake.sizes = lambda: [2**30, 3, 2**30]
    x_fake.dim = 3

    kernel = build_kernel()
    with pytest.raises(Exception):
        _ = kernel.forward(x_fake, 1)

if __name__ == "__main__":
    pytest.main([__file__])
