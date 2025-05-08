
import pytest
import torch
from torch.utils.cpp_extension import load

# A helper function to build the CUDA extension from kernel.cu
def build_kernel():
    module = load(
        name="kl_div_combined",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Incorrect normalization factor.
# We prepare a small tensor (batch size 2, 4 elements per row) so that n != batch_size.
def test_normalization_factor():
    # Create two 2x4 tensors that sum to one along dim=-1
    predictions = torch.tensor([[0.2, 0.3, 0.5, 0.0],
                                [0.1, 0.1, 0.8, 0.0]], device='cuda', dtype=torch.float32)
    predictions = predictions / predictions.sum(dim=1, keepdim=True)
    targets = torch.tensor([[0.3, 0.3, 0.4, 0.0],
                            [0.2, 0.2, 0.6, 0.0]], device='cuda', dtype=torch.float32)
    targets = targets / targets.sum(dim=1, keepdim=True)
    
    # Compute reference using PyTorch function which divides by the batch size (i.e. 2)
    ref = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    
    mod = build_kernel()
    # Our kernel expects log_predictions already computed.
    out = mod.forward(torch.log(predictions), targets)
    
    # Since our kernel divides by total number of elements (8 in this case) the result will be different.
    # Therefore, the test asserts that the computed and reference outputs are not close.
    assert not torch.allclose(out, ref, atol=1e-5), \
           f"Normalization issue not detected: kernel output {out.item()} matches reference {ref.item()}"

# Issue 2: Non-contiguous input causing misaligned vectorized loads.
def test_non_contiguous_input():
    # Create a 2D tensor then transpose it, which typically makes it non-contiguous.
    base = torch.randn(8, 4, device="cuda", dtype=torch.float32)
    predictions = base.t().softmax(dim=-1)
    targets = base.t().softmax(dim=-1)
    
    mod = build_kernel()
    # Since our kernel assumes contiguous memory, we expect an error when passed non-contiguous data.
    with pytest.raises(RuntimeError):
        _ = mod.forward(torch.log(predictions), targets)

# Issue 3: Wrong tensor type (not float32).
def test_wrong_tensor_type():
    # Create two tensors of type double.
    predictions = torch.randn(8, 4, device="cuda", dtype=torch.float64).softmax(dim=-1)
    targets = torch.randn(8, 4, device="cuda", dtype=torch.float64).softmax(dim=-1)
    
    mod = build_kernel()
    # The kernel only supports float and uses reinterpret_cast to reinterpret data as float4.
    # Hence, passing double tensors should lead to a runtime error.
    with pytest.raises(RuntimeError):
        _ = mod.forward(torch.log(predictions), targets)

# Issue 4: Misaligned tensor due to pointer offset.
def test_misaligned_tensor():
    # Allocate a larger tensor and take a slice such that its data pointer is offset
    base = torch.randn(20, device="cuda", dtype=torch.float32)
    # Create a sliced view starting from element 1, which is typically misaligned for vectorized loads.
    misaligned = base[1:17]  # This tensor has 16 elements but is likely not 16-byte aligned.
    # Reshape into a 4x4 matrix (contiguous but misaligned at the start)
    predictions = misaligned.view(4, 4).softmax(dim=-1)
    targets = predictions.clone()
    
    mod = build_kernel()
    # While the kernel may run without a hard exception, the vectorized load might produce a result
    # different from the expected CPU computation.
    out = mod.forward(torch.log(predictions), targets)
    ref = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    
    assert not torch.allclose(out, ref, atol=1e-5), \
           f"Misalignment issue not detected: kernel output {out.item()} matches reference {ref.item()}"

