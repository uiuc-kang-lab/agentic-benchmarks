
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Utility function to compile and load the kernel extension, assuming kernel.cu is in the same directory.
def build_kernel():
    src_path = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="mse_cuda_unroll",
        sources=[src_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 2: Noncontiguous tensor test.
def test_non_contiguous_tensors():
    # Create contiguous tensors first and then make them noncontiguous via transposition or slicing.
    A = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    B = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    # Make them noncontiguous by selecting a slice in the last dimension.
    A_nc = A[:, ::2]
    B_nc = B[:, ::2]
    # Compute ground truth MSE using PyTorch’s own operations
    mse_ref = torch.mean((A_nc - B_nc) ** 2)
    # Run the CUDA kernel (which assumes contiguous inputs)
    module = build_kernel()
    mse_cuda = module.forward(A_nc, B_nc)
    torch.cuda.synchronize()
    # Expect the kernel result to differ from the correct result because the underlying memory layout is not as assumed.
    assert not torch.allclose(mse_cuda, mse_ref, atol=1e-6), \
        "The kernel did not reveal an issue with noncontiguous inputs."

# Issue 3: Pragma unroll usage.
# While we cannot force a compile-time error from Python, we can trigger a test case that uses unrolling.
# For example, run on a tensor with size that is not a multiple of the unroll factor.
def test_tensor_size_not_multiple_of_unroll():
    # Choose a tensor size that is deliberately not a multiple of (BLOCK_SIZE * UNROLL_FACTOR).
    # Using 131, for instance.
    n = 131  
    A = torch.randn(n, device="cuda", dtype=torch.float32)
    B = torch.randn(n, device="cuda", dtype=torch.float32)
    mse_ref = torch.mean((A-B)**2)
    module = build_kernel()
    mse_cuda = module.forward(A, B)
    torch.cuda.synchronize()
    # We expect a discrepancy if the kernel launch config (grid size not accounting for unroll factor)
    # does not cover all elements properly.
    assert not torch.allclose(mse_cuda, mse_ref, atol=1e-6), \
        f"Kernel result (MSE = {mse_cuda.item()}) unexpectedly agrees with reference when tensor size is not a multiple of the unroll factor."

# Issue 4: Grid size miscalculation due to unroll factor.
def test_grid_size_issue():
    # Create an array with a size that forces several unroll iterations.
    # The total number of elements is not a multiple of BLOCK_SIZE*UNROLL_FACTOR.
    # (We know BLOCK_SIZE is 256 and UNROLL_FACTOR is 4 in the code.)
    n = 1000  # 1000 is not a multiple of 256*4 = 1024.
    A = torch.randn(n, device="cuda", dtype=torch.float32)
    B = torch.randn(n, device="cuda", dtype=torch.float32)
    mse_ref = torch.mean((A - B) ** 2)
    module = build_kernel()
    mse_cuda = module.forward(A, B)
    torch.cuda.synchronize()
    # In a correct implementation, the average MSE should be computed precisely.
    # If the grid size is miscalculated, some elements might be skipped or double-counted.
    # We check if the relative error is large.
    rel_err = abs(mse_cuda.item() - mse_ref.item()) / (mse_ref.item() + 1e-10)
    assert rel_err > 1e-3, \
        f"Relative error {rel_err} is too small; expected a discrepancy if grid size is miscomputed."

# Issue 5: Atomic add on double not supported on older architectures.
# We check the compute capability and skip this test if we are on a new enough GPU.
def test_atomic_add_double_support():
    capability = torch.cuda.get_device_capability()
    if capability[0] < 6:
        # On older GPUs we expect the kernel to fail (or produce an error)
        A = torch.randn(128, device="cuda", dtype=torch.float32)
        B = torch.randn(128, device="cuda", dtype=torch.float32)
        module = build_kernel()
        with pytest.raises(RuntimeError):
            _ = module.forward(A, B)
    else:
        pytest.skip("Test not applicable: GPU supports double atomicAdd.")

# Issue 1: Indexing overflow when num_elements > INT_MAX.
# It is not practical to allocate more than INT_MAX elements in a test.
# Instead, we simulate by monkey-patching the numel() method of the input tensors.
class FakeLargeTensor:
    def __init__(self, tensor, fake_numel):
        self.tensor = tensor
        self.fake_numel = fake_numel

    def __torch_function__(self, func, types, args=(), kwargs=None):
        return func(*args, **kwargs)

    def data_ptr(self):
        return self.tensor.data_ptr()

    @property
    def device(self):
        return self.tensor.device

    @property
    def dtype(self):
        return self.tensor.dtype

    def numel(self):
        return self.fake_numel

def test_indexing_overflow_simulation():
    # Create a modest tensor but override its numel() to simulate a huge tensor.
    A = torch.randn(128, device="cuda", dtype=torch.float32)
    B = torch.randn(128, device="cuda", dtype=torch.float32)
    fake_num = (1 << 31) + 10  # Greater than INT_MAX
    A_fake = FakeLargeTensor(A, fake_num)
    B_fake = FakeLargeTensor(B, fake_num)

    module = build_kernel()
    # Since the kernel uses the fake huge numel, the loop bounds will be incorrect.
    # We expect either a wrong result or a runtime error because the indexing loop is not truly iterating over huge memory.
    with pytest.raises(Exception):
        _ = module.forward(A_fake, B_fake)

# Issue 6: Using half precision (float16) input.
# The kernel uses AT_DISPATCH_FLOATING_TYPES which might not include at::kHalf.
def test_half_precision():
    A = torch.randn(128, 4096, device="cuda", dtype=torch.float16)
    B = torch.randn(128, 4096, device="cuda", dtype=torch.float16)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel dispatch will likely raise because float16 is not handled
        _ = module.forward(A, B)
