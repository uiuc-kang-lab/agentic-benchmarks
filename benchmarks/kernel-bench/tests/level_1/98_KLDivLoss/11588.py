
import pytest
import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel():
    # This builds the CUDA extension from the file kernel.cu.
    module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Incorrect KL divergence computation.
def test_incorrect_kl_computation():
    module = build_kernel()
    # Create a simple input tensor where we can compute the expected value.
    # Using 1 element to simplify the analysis.
    predictions = torch.tensor([0.8], dtype=torch.float32, device="cuda").unsqueeze(0)
    targets = torch.tensor([0.2], dtype=torch.float32, device="cuda").unsqueeze(0)
    # Using softmax so that predictions sum to 1. Here it is already a 1D distribution.
    predictions = F.softmax(predictions, dim=-1)
    targets = F.softmax(targets, dim=-1)
    
    # The reference computation from PyTorch:
    # Expected: loss = target*(log(target)-log(pred)) averaged over batch (batchmean)
    ref_loss = F.kl_div(torch.log(predictions), targets, reduction="batchmean")
    
    # Kernel computation
    loss_kernel = module.forward(torch.log(predictions), targets)
    
    # Since the kernel uses an incorrect formula, the result should not match the reference.
    # We expect the relative error to be large.
    rel_error = torch.abs(loss_kernel - ref_loss) / (torch.abs(ref_loss) + 1e-6)
    assert rel_error.item() > 0.1, f"Kernel KL divergence seems correct; expected error due to wrong formula. rel_error={rel_error.item()}"

# Issue 2: Wrong normalization factor.
def test_normalization_factor():
    module = build_kernel()
    # Create a batch with more than one distribution.
    batch = 16
    dim = 10
    predictions = torch.randn(batch, dim, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch, dim, device="cuda").softmax(dim=-1)
    
    # Reference using batchmean reduction divides by batch size
    ref_loss = F.kl_div(torch.log(predictions), targets, reduction="batchmean")
    
    loss_kernel = module.forward(torch.log(predictions), targets)
    
    # The kernel divides the sum by total number of elements (batch*dim) which is different.
    # Therefore the kernel result should differ from the reference.
    diff = torch.abs(loss_kernel - ref_loss).item()
    assert diff > 1e-6, f"Kernel normalization factor appears correct (difference {diff}), but it should differ due to division by total elements."

# Issue 3: Lack of type checks (input tensor type is not float32)
def test_input_tensor_type():
    module = build_kernel()
    # Create double precision input tensors instead of float32.
    batch = 8
    dim = 5
    predictions = torch.randn(batch, dim, device="cuda", dtype=torch.float64).softmax(dim=-1)
    targets = torch.randn(batch, dim, device="cuda", dtype=torch.float64).softmax(dim=-1)
    
    # Here we expect the kernel to fail (or produce garbage) because it expects float32.
    with pytest.raises(RuntimeError):
        # The kernel is launched with data_ptr<float> so passing double will cause an error.
        module.forward(torch.log(predictions), targets)
    
# Issue 4: Lack of contiguous memory checks (non-contiguous inputs).
def test_non_contiguous_input():
    module = build_kernel()
    batch = 16
    dim = 32
    predictions = torch.randn(batch, dim, device="cuda", dtype=torch.float32)
    targets = torch.randn(batch, dim, device="cuda", dtype=torch.float32)
    
    # Make the tensors non-contiguous by transposing (if possible) or slicing with a step.
    predictions = predictions.t()  # Transpose makes it non-contiguous (dim becomes batch)
    targets = targets.t()
    
    # The kernel uses data_ptr assuming contiguous memory.
    # We expect that the wrong memory layout will lead to an incorrect result.
    loss_kernel = module.forward(torch.log(predictions), targets)
    # As a reference we force contiguous tensors before using PyTorch's F.kl_div.
    ref_loss = F.kl_div(torch.log(predictions.contiguous()), targets.contiguous(), reduction="batchmean")
    diff = torch.abs(loss_kernel - ref_loss).item()
    assert diff > 1e-6, f"Kernel handled non-contiguous input correctly, but it was supposed to assume contiguous memory. Difference: {diff}"
