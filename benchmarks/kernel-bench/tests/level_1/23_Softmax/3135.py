
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_softmax_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Misaligned memory access when using vectorized loads/stores.
def test_misaligned_data():
    # Create a tensor that is 2D and whose num_features is divisible by 4.
    batch_size = 4
    num_features = 128  # Divisible by 4.
    # Allocate a larger tensor and then take a slice to force misalignment.
    base = torch.randn(batch_size, num_features + 1, device='cuda', dtype=torch.float32)
    # Taking all rows and columns from index 1 onward: this may misalign the pointer.
    x = base[:, 1:]
    # x now has shape (batch_size, num_features) but its underlying data pointer
    # is likely not 16-byte aligned.
    cuda_module = build_kernel()
    # The kernel will take the vectorized code path because num_features is divisible by 4.
    # This test is intended to trigger misaligned accesses.
    with pytest.warns(UserWarning):
        y = cuda_module.forward(x)
    # We only check that the output has the correct shape.
    assert y.shape == (batch_size, num_features), "Output shape mismatch in misaligned data test."

# Issue 2: BlockDim reduction assumptions when blockDim.x is not a multiple of 32.
def test_nonwarp_aligned_block():
    # Although the kernel is launched using a fixed THREADS_PER_BLOCK=256 (a multiple of 32),
    # we simulate a problematic scenario by calling the reduction code with settings that
    # might occur in a more general context.
    #
    # We simulate this indirectly by using a tensor whose number of features is small so that only
    # a few threads are active. In that case, the implicit assumption that a full warp participates
    # in the reduction can break.
    batch_size = 2
    num_features = 10  # Small size; reduction may not be robust if threads < 32 are active.
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float32)
    cuda_module = build_kernel()
    # Although the kernel code always launches 256 threads per block,
    # with a small work size the later iterations may operate on inactive threads.
    y = cuda_module.forward(x)
    # Compute a reference softmax for comparison.
    x_cpu = x.cpu()
    y_ref = torch.softmax(x_cpu, dim=1)
    # Allow a looser tolerance as reduction may accumulate rounding differences.
    assert torch.allclose(y.cpu(), y_ref, atol=1e-4), "Softmax output differs when using non-warp-multiple work sizes."

# Issue 3: Limited support for only 2D contiguous inputs.
def test_non_2d_input():
    # Pass a 3D tensor (e.g., batch_size x height x width) to trigger the input-dimension check.
    x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The forward function has a TORCH_CHECK for 2D inputs.
        cuda_module.forward(x)

if __name__ == "__main__":
    pytest.main([__file__])
