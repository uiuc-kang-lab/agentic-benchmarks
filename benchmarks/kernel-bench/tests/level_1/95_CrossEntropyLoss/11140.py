
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="cross_entropy_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test for out-of-bound target index.
def test_out_of_bound_target():
    # Use a valid batch_size but deliberately set one target index out of range.
    batch_size = 256
    num_classes = 10
    # Generate predictions tensor
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    # Generate valid targets, then set one index to an invalid value.
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    targets[0] = num_classes  # Setting an out-of-bound index.
    
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel does not do explicit bounds checking; this memory error should surface.
        cuda_module.forward(predictions, targets)
    torch.cuda.synchronize()

# Issue 2: Test for non-contiguous predictions tensor.
def test_non_contiguous_predictions():
    batch_size = 128
    num_classes = 10
    
    # Create a tensor with extra dimensions then transpose to force non-contiguity.
    x = torch.randn(1, batch_size, num_classes, device="cuda", dtype=torch.float32)
    predictions = x.squeeze(0).t()  # This makes the tensor non-contiguous.
    # To match the expected shape, transpose back.
    predictions = predictions.t()
    
    # Create valid targets.
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    
    cuda_module = build_kernel()
    # The kernel assumes a contiguous tensor; non-contiguity might cause wrong results.
    loss_cuda = cuda_module.forward(predictions, targets)
    # Compare with PyTorch's built-in result.
    loss_ref = torch.nn.functional.cross_entropy(predictions.contiguous(), targets)
    
    # We do not expect the correct numerical result; the assertion should fail.
    with pytest.raises(AssertionError):
        assert torch.allclose(loss_cuda, loss_ref, atol=1e-4), "Loss mismatch with non-contiguous input"
    torch.cuda.synchronize()

# Issue 3: Test for num_classes that is not a multiple of 4 (in particular, less than 4).
def test_num_classes_not_multiple_of_four():
    # Use a small number of classes that is not divisible by 4.
    batch_size = 64
    num_classes = 3  # less than 4
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    
    cuda_module = build_kernel()
    loss_cuda = cuda_module.forward(predictions, targets)
    loss_ref = torch.nn.functional.cross_entropy(predictions, targets)
    
    # For a correct kernel, the loss values should match within a tolerance.
    # If the unrolling causes issues, the values may differ.
    assert torch.allclose(loss_cuda, loss_ref, atol=1e-4), (
        f"Kernel output ({loss_cuda}) differs from reference output ({loss_ref}) when num_classes={num_classes}"
    )
    torch.cuda.synchronize()
    
if __name__ == '__main__':
    pytest.main([__file__])
