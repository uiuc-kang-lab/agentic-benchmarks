
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel():
    return load(
        name="kl_div_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Test case 1: Incorrect formula computation.
# We'll compare the kernel output to the expected PyTorch computation.
# Since the kernel omits the targets * log(targets) term, the outputs will differ.
def test_incorrect_formula():
    # Create input distributions that are well-behaved
    batch_size = 4
    input_shape = (16,)
    predictions = torch.randn(batch_size, *input_shape, device='cuda')
    predictions = torch.softmax(predictions, dim=-1)
    targets = torch.randn(batch_size, *input_shape, device='cuda')
    targets = torch.softmax(targets, dim=-1)
    
    # Compute using PyTorch's built-in F.kl_div (which does targets*(log(targets)-log_predictions) / batch_size)
    ref = F.kl_div(torch.log(predictions), targets, reduction='batchmean')
    
    cuda_kernel = build_kernel()
    # The kernel expects the inputs to be of type float32.
    log_predictions = torch.log(predictions).contiguous()
    targets_contig = targets.contiguous()
    kernel_out = cuda_kernel.forward(log_predictions, targets_contig)
    
    # Because of the missing term, the kernel output will be different
    assert not torch.allclose(kernel_out, ref, atol=1e-5), (
        "Test failed: Kernel output unexpectedly matches the PyTorch result. "
        "This indicates the missing targets*log(targets) term is not being exposed."
    )

# Test case 2: Incorrect normalization factor.
# We'll run a case where total elements n is not equal to batch_size,
# and compare the scaling difference.
def test_incorrect_normalization():
    # Use a batch size and input shape such that n >> batch_size
    batch_size = 8
    input_shape = (1024,)
    predictions = torch.randn(batch_size, *input_shape, device='cuda')
    predictions = torch.softmax(predictions, dim=-1)
    targets = torch.randn(batch_size, *input_shape, device='cuda')
    targets = torch.softmax(targets, dim=-1)
    
    cuda_kernel = build_kernel()
    log_predictions = torch.log(predictions).contiguous()
    targets_contig = targets.contiguous()
    kernel_out = cuda_kernel.forward(log_predictions, targets_contig)
    
    # The PyTorch reference divides by batch size, so its scale is different.
    ref = F.kl_div(torch.log(predictions), targets, reduction='batchmean')
    
    # Since the kernel divides by n (total elements) not batch size, the outputs should differ.
    assert not torch.allclose(kernel_out, ref, atol=1e-5), (
        "Test failed: Kernel normalization appears correct, but it should differ from batchmean normalization."
    )

# Test case 3: Mismatched input sizes.
# The kernel does no bounds checking, so if the number of elements in predictions
# and targets differ, it may perform out-of-bound accesses.
def test_mismatched_input_sizes():
    batch_size = 4
    input_shape_pred = (32,)
    input_shape_target = (16,)  # deliberately different
    predictions = torch.randn(batch_size, *input_shape_pred, device='cuda')
    predictions = torch.softmax(predictions, dim=-1)
    targets = torch.randn(batch_size, *input_shape_target, device='cuda')
    targets = torch.softmax(targets, dim=-1)
    
    cuda_kernel = build_kernel()
    log_predictions = torch.log(predictions).contiguous()
    targets_contig = targets.contiguous()
    
    # Expect runtime error due to out-of-bound access.
    with pytest.raises(RuntimeError):
        _ = cuda_kernel.forward(log_predictions, targets_contig)
        torch.cuda.synchronize()

# Test case 4: Input tensor type mismatch (e.g. using float64 instead of float32).
def test_input_tensor_type():
    batch_size = 4
    input_shape = (64,)
    predictions = torch.randn(batch_size, *input_shape, device='cuda', dtype=torch.float64)
    predictions = torch.softmax(predictions, dim=-1)
    targets = torch.randn(batch_size, *input_shape, device='cuda', dtype=torch.float64)
    targets = torch.softmax(targets, dim=-1)
    
    cuda_kernel = build_kernel()
    log_predictions = torch.log(predictions).contiguous()
    targets_contig = targets.contiguous()
    
    # Expect an error or incorrect behavior due to type mismatch.
    with pytest.raises(RuntimeError):
        _ = cuda_kernel.forward(log_predictions, targets_contig)
        torch.cuda.synchronize()

# Test case 5: Lack of kernel launch error checking.
# We attempt to trigger an error by forcing an illegal memory access.
def test_kernel_launch_error_checking():
    # Create an empty tensor which will lead to n==0 and potentially misbehave.
    predictions = torch.empty((0,), device='cuda', dtype=torch.float32)
    targets = torch.empty((0,), device='cuda', dtype=torch.float32)
    
    cuda_kernel = build_kernel()
    
    # Launching kernel with zero elements might cause an error in the kernel’s execution
    # because the grid-stride loop will not run and later atomic adds might be misbehaving.
    # Alternatively, if the kernel silently does nothing, we check that the output is zero.
    try:
        out = cuda_kernel.forward(predictions, targets)
        torch.cuda.synchronize()
        # In our kernel, output is divided by n (which is 0). This might produce NaN or error.
        assert torch.isnan(out).any() or torch.isinf(out).any(), (
            "Test failed: Kernel launch with zero elements should result in undefined output."
        )
    except RuntimeError:
        # If a runtime error is raised due to division by zero or other illegal access, that is acceptable.
        pass
