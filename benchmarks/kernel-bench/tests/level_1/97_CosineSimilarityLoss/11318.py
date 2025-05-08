
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="cosine_similarity_loss_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger out-of-bound access via grid mis-configuration.
# We simulate this by manually launching the kernel with an extra block.
def test_out_of_bound_block_index():
    module = build_kernel()
    # Create valid inputs for one row.
    N = 1
    D = 4096
    predictions = torch.randn(N, D, device="cuda", dtype=torch.float32)
    targets = torch.randn(N, D, device="cuda", dtype=torch.float32)
    # Allocate output tensor.
    output = torch.zeros(1, device="cuda", dtype=torch.float32)
    
    # Prepare kernel launch parameters intentionally using grid size (N+1)
    block_size = 256
    numWarps = (block_size + 31) // 32
    shared_mem = numWarps * 3 * 4  # 4 bytes per float

    # Manually launch kernel with an extra block.
    # The extra block will set row = blockIdx.x == 1 when N == 1.
    stream = torch.cuda.current_stream().cuda_stream
    module.cosine_similarity_loss_kernel[
        (N + 1,), block_size, shared_mem, stream
    ](
        predictions.data_ptr(), targets.data_ptr(), output.data_ptr(), N, D
    )
    torch.cuda.synchronize()
    # If out-of-bound is not handled, memory corruption may occur.
    # We check that the accumulated loss does not exceed the maximum possible loss.
    # (This is a heuristic test meant to detect anomalous results.)
    loss = output.item() / (N if N>0 else 1)
    # For cosine similarity loss, the expected per-row loss is in [0,2]
    assert 0.0 <= loss <= 2.0, f"Accumulated loss {loss} is out-of-bound, indicating a possible OOB access."

# Test case 2: Non-contiguous inputs.
def test_non_contiguous_inputs():
    module = build_kernel()
    N = 128
    D = 4096
    # Create contiguous tensors and then make non-contiguous versions via transpose
    a = torch.randn(N, D, device="cuda", dtype=torch.float32)
    b = torch.randn(N, D, device="cuda", dtype=torch.float32)
    # Making them non-contiguous by unsqueeze and squeeze along a new dimension
    a_nc = a.unsqueeze(0).squeeze(0)
    b_nc = b.unsqueeze(0).squeeze(0)
    # Confirm non-contiguity
    assert not a_nc.is_contiguous(), "Test setup error: tensor 'a_nc' should be non-contiguous."
    # Run the forward function (which calls the kernel)
    loss = module.forward(a_nc, b_nc)
    # Expected loss according to PyTorch’s own computation:
    cos_sim = torch.nn.functional.cosine_similarity(a_nc, b_nc, dim=1)
    loss_ref = torch.mean(1 - cos_sim).item()
    loss_cuda = loss.item()
    # Because the kernel assumes contiguous layout, non-contiguous inputs may lead to wrong results.
    # So we expect the loss to differ from the reference.
    assert abs(loss_cuda - loss_ref) > 1e-3, (
        f"Kernel did not trigger an error with non-contiguous inputs: loss {loss_cuda} vs reference {loss_ref}"
    )

# Test case 3: Type support – pass double (float64) tensors.
def test_input_tensor_type():
    module = build_kernel()
    N = 128
    D = 4096
    predictions = torch.randn(N, D, device="cuda", dtype=torch.float64)
    targets = torch.randn(N, D, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="predictions must be float32"):
        # The TORCH_CHECK in the forward function should trigger an error.
        module.forward(predictions, targets)

# Test case 4: Input tensor dimensions – pass a tensor with >2 dimensions.
def test_input_tensor_dimension():
    module = build_kernel()
    # Create a 3D tensor which is not supported.
    predictions = torch.randn(10, 128, 4096, device="cuda", dtype=torch.float32)
    targets = torch.randn(10, 128, 4096, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="predictions must be 2D"):
        module.forward(predictions, targets)
