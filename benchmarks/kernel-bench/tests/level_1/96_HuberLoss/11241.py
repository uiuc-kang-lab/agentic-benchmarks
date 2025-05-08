
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# A helper reference function using PyTorch's own smooth_l1_loss.
def smooth_l1_loss_reference(predictions, targets):
    diff = predictions - targets
    abs_diff = diff.abs()
    loss = torch.where(
        abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5
    )
    return loss.mean()

# 1. Test for unsupported input type (issue 1)
def test_input_tensor_type_error():
    # Creating double tensors even though the kernel expects float32.
    predictions = torch.randn(128, 4096, dtype=torch.float64, device="cuda")
    targets = torch.randn(128, 4096, dtype=torch.float64, device="cuda")
    module = build_kernel()
    # The kernel does not check input type, so it will use data_ptr<float>()
    # This leads to misinterpreting the memory and producing an incorrect result.
    # Although it might not throw an exception immediately, we check that the output
    # from the CUDA kernel deviates significantly from the reference.
    loss_kernel = module.forward(predictions.float(), targets.float())
    loss_ref = smooth_l1_loss_reference(predictions.float(), targets.float())
    # Here, instead of converting back to double, we purposely cast the original double
    # tensors to float and then compare so that a misuse of types is revealed.
    # Run the kernel with double data reinterpreted as float.
    predictions_bad = predictions  # double tensor, wrong type for kernel
    targets_bad = targets
    with pytest.warns(UserWarning):
        # We expect that the kernel output is not close to the correct result because
        # of the misinterpretation of data (the test assumes that wrong types lead to wrong behavior).
        loss_bad = module.forward(predictions_bad, targets_bad)
    # Compare to reference computed with proper float conversion:
    loss_correct = smooth_l1_loss_reference(predictions_bad.float(), targets_bad.float())
    assert not torch.allclose(loss_bad, loss_correct, atol=1e-3), \
        "Kernel accepts non-float32 data without error but produces a correct result, which is unexpected."

# 2. Test for non‐multiple‐of‐warp block sizes (issue 2)
def test_non_multiple_warp_block():
    # Create a tensor with a very small number of elements so that the grid will launch a full block
    # even though only a few threads actually have work. For example, if we create only 20 elements,
    # block_size is still hard coded to 256. Many threads are idle but still participate in shuffle-reduction.
    n = 20
    predictions = torch.randn(n, device="cuda", dtype=torch.float32)
    targets = torch.randn(n, device="cuda", dtype=torch.float32)
    module = build_kernel()
    loss_kernel = module.forward(predictions, targets)
    loss_ref = smooth_l1_loss_reference(predictions, targets)
    # Due to the reduction assumption, the result may be incorrect.
    assert not torch.allclose(loss_kernel, loss_ref, atol=1e-5), \
        "Kernel warp-level reduction appears correct with non-multiple-of-warp block size, but an error was expected."

# 3. Test for empty input tensor (issue 3)
def test_empty_input():
    # Create empty tensors. This will result in n_elements = 0 in the kernel.
    # Division by zero in the kernel should produce NaN in the output.
    predictions = torch.tensor([], device="cuda", dtype=torch.float32)
    targets = torch.tensor([], device="cuda", dtype=torch.float32)
    module = build_kernel()
    loss_kernel = module.forward(predictions, targets)
    # The expected behavior (although undefined) is to get a NaN.
    assert torch.isnan(loss_kernel).item(), \
        "For empty input, the kernel did not produce NaN (division by zero) as expected."

# 4. Test for non-CUDA input to trigger TORCH_CHECK failure (and indirectly the lack of error checking)
def test_non_cuda_input():
    # Create CPU tensors and pass them to the kernel.
    predictions = torch.randn(128, 4096, dtype=torch.float32, device="cpu")
    targets = torch.randn(128, 4096, dtype=torch.float32, device="cpu")
    module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        module.forward(predictions, targets)
    assert "Inputs must be CUDA tensors" in str(excinfo.value)

if __name__ == "__main__":
    pytest.main([__file__])
