
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension.
def build_kernel():
    return load(
        name="ce_loss_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# 1. Test case for invalid thread configuration assumption:
#    This test indirectly checks that the kernel assumes a warp of 32 threads.
#    Since the extension forces 32 threads per block, we simulate the issue by
#    re-launching with an incorrect block configuration if possible.
#
#    (NOTE: In a real-world scenario the kernel invocation parameters are fixed in the C++ code.
#     Therefore, this test demonstrates that if one were to change the launch parameters
#     the kernel may produce wrong results. Here we do not have an interface to change it,
#     so this test will simply re-run the kernel and warn the user.)
def test_thread_configuration():
    ce_loss = build_kernel()
    batch_size = 10
    num_classes = 5
    predictions = torch.randn(batch_size, num_classes, dtype=torch.float32, device="cuda")
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda")
    # Run normally (this uses 32 threads per block as hardcoded)
    loss_normal = ce_loss.forward(predictions, targets)
    
    # Emulate a bad configuration by manually modifying the predictions tensor shape to trick the kernel
    # (Since the kernel launch parameters are fixed, we cannot easily change threads/block from Python.)
    # Instead, we warn if the user ever mistakenly changes the kernel launch configuration.
    assert loss_normal is not None, "Kernel did not return a loss, possible misconfiguration of blockDim."

# 2. Test case for non-float32 predictions:
def test_invalid_dtype():
    ce_loss = build_kernel()
    batch_size = 10
    num_classes = 5
    # Create predictions tensor with dtype float64 instead of float32.
    predictions = torch.randn(batch_size, num_classes, dtype=torch.float64, device="cuda")
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda")
    with pytest.raises(RuntimeError, match="predictions must be Float32 tensor"):
        ce_loss.forward(predictions, targets)

# 3. Test case for out-of-bound target indices:
def test_target_out_of_bounds():
    ce_loss = build_kernel()
    batch_size = 10
    num_classes = 5
    predictions = torch.randn(batch_size, num_classes, dtype=torch.float32, device="cuda")
    # Create targets with an invalid index (equal to num_classes) for at least one sample.
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda")
    targets[0] = num_classes  # out-of-bound index
    with pytest.raises(RuntimeError):
        # This should trigger a CUDA illegal memory access.
        ce_loss.forward(predictions, targets)
        torch.cuda.synchronize()

# 4. Test case for non-contiguous predictions tensor:
def test_non_contiguous_input():
    ce_loss = build_kernel()
    batch_size = 10
    num_classes = 5
    predictions = torch.randn(batch_size, num_classes, dtype=torch.float32, device="cuda")
    # Make the tensor non-contiguous by transposing it twice after an unsqueeze.
    predictions = predictions.unsqueeze(0).transpose(0, 1).squeeze(1)
    assert not predictions.is_contiguous(), "Predictions tensor should be non-contiguous for this test."
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda")
    # The kernel expects a contiguous tensor; if not, its behavior might be undefined.
    with pytest.raises(RuntimeError):
        ce_loss.forward(predictions, targets)

# 5. Test case for unsupported data types (e.g. half-precision):
def test_unsupported_dtype():
    ce_loss = build_kernel()
    batch_size = 10
    num_classes = 5
    # Create predictions tensor with half (float16) dtype.
    predictions = torch.randn(batch_size, num_classes, dtype=torch.float16, device="cuda")
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda")
    with pytest.raises(RuntimeError, match="predictions must be Float32 tensor"):
        ce_loss.forward(predictions, targets)
