
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test kernel's assumption of float32 inputs.
def test_input_tensor_type():
    # Create inputs as double (float64) which the kernel does not support.
    batch_size = 128
    input_shape = (4096,)
    # Using softmax so that values sum to one.
    predictions = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float64).softmax(dim=-1)
    targets = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float64).softmax(dim=-1)
    my_kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # This should error because the kernel expects float32 pointers.
        output = my_kernel.forward(predictions, targets)
        torch.cuda.synchronize()

# Issue 2: Test the normalization mismatch.
def test_incorrect_normalization():
    # Create inputs as in the python Model forward
    batch_size = 4  # small batch so expected difference is visible
    input_shape = (1024,)
    predictions = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float32).softmax(dim=-1)
    # Compute reference using PyTorch's F.kl_div.
    ref = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    # Get kernel output
    my_kernel = build_kernel()
    out = my_kernel.forward(torch.log(predictions), targets)
    torch.cuda.synchronize()
    # Since the kernel divides by total number of elements rather than batch size,
    # the normalization factor is off. In a well‐behaved kernel these should be close.
    # We check that the error is indeed above a tolerance threshold.
    tol = 1e-3
    diff = (out.item() - ref.item())
    assert abs(diff) > tol, f"Expected normalization error, but got diff={diff}"

# Issue 3: Test that non-contiguous inputs are not handled correctly.
def test_non_contiguous_input():
    batch_size = 128
    input_shape = (4096,)
    # Create contiguous predictions and targets first
    predictions = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float32).softmax(dim=-1)
    # Make non-contiguous versions by transposing a dummy extra dimension.
    predictions_nc = predictions.unsqueeze(2).transpose(1,2).squeeze(2)
    targets_nc = targets.unsqueeze(2).transpose(1,2).squeeze(2)
    assert not predictions_nc.is_contiguous(), "Predictions should be non-contiguous for this test"
    my_kernel = build_kernel()
    # Depending on hardware behavior the kernel might compute a wrong result.
    # Here we check that the output deviates from the contiguous version.
    out_nc = my_kernel.forward(torch.log(predictions_nc), targets_nc)
    out_c  = my_kernel.forward(torch.log(predictions), targets)
    torch.cuda.synchronize()
    tol = 1e-5
    diff = abs(out_nc.item() - out_c.item())
    assert diff > tol, f"Expected difference between contiguous and non-contiguous inputs, but diff={diff}"

if __name__ == '__main__':
    pytest.main([__file__])
