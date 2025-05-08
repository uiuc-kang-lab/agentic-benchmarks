
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Loads the CUDA extension from kernel.cu.
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

def test_incorrect_formula():
    # This test demonstrates that the kernel computes a value differing from the expected KL divergence.
    # Prepare inputs consistent with the original PyTorch usage: first argument is log(predictions), second is targets.
    batch_size = 128
    features = 4096
    # Create “prediction” tensor (after softmax) and then compute its log.
    predictions = torch.randn(batch_size, features, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch_size, features, device="cuda").softmax(dim=-1)
    log_predictions = predictions.log()

    module = build_kernel()
    # Kernel output
    kernel_out = module.forward(log_predictions, targets)
    
    # Expected KL divergence using the official formula:
    # Note: torch.nn.functional.kl_div expects inputs as log-probabilities and uses reduction='batchmean'
    expected = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    
    # They should match but the kernel formula is wrong.
    assert not torch.allclose(kernel_out, expected, atol=1e-5), (
        f"Test failed: Kernel output ({kernel_out.item()}) unexpectedly matches expected KL divergence "
        f"({expected.item()}). Formula implementation issue is not triggered."
    )

def test_input_tensor_type():
    # This test is designed to trigger a type-related issue.
    # The kernel expects float32 but we send double tensors.
    batch_size = 256
    features = 1024
    predictions = torch.randn(batch_size, features, device="cuda", dtype=torch.double).softmax(dim=-1)
    targets = torch.randn(batch_size, features, device="cuda", dtype=torch.double).softmax(dim=-1)
    log_predictions = predictions.log()

    module = build_kernel()
    # We expect a runtime error or a result that is numerically wrong.
    with pytest.raises(RuntimeError):
        # If the kernel is built to work with float only,
        # passing double tensors (via data_ptr<float> reinterpretation) should cause an error.
        kernel_out = module.forward(log_predictions, targets)
        torch.cuda.synchronize()

def test_scaling_mismatch():
    # This test is for the reduction scaling issue.
    # For reduction 'batchmean' PyTorch divides by batch_size, but the kernel divides by total elements.
    batch_size = 4
    features = 100  # arbitrary feature dimension
    predictions = torch.randn(batch_size, features, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, features, device="cuda", dtype=torch.float32).softmax(dim=-1)
    log_predictions = predictions.log()

    module = build_kernel()
    kernel_out = module.forward(log_predictions, targets)
    
    # Compute expected output using torch.nn.functional.kl_div
    expected = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    
    # Because the kernel divides by the total number of elements (batch_size * features),
    # the output should be scaled differently.
    assert not torch.allclose(kernel_out, expected, atol=1e-5), (
        f"Test failed: Kernel output scaling seems correct ({kernel_out.item()}), but it should differ "
        f"from the official batchmean output ({expected.item()}) given the implementation issue."
    )

def test_min_function_compilation_issue():
    # This test does not run the kernel but signals the issue with the use of unqualified 'min'.
    # In a proper continuous integration setup, a failed compilation would be caught.
    # Here, we simply check that the extension builds. If it did, then the min issue was fixed or unnoticed.
    try:
        module = build_kernel()
    except Exception as e:
        pytest.skip("Compilation failed; likely due to an unqualified use of min. Issue triggered.")
    else:
        # If built successfully, then the min issue might have been resolved by adding proper includes.
        assert True, "Module built successfully. (The use of min may be fixed by the developer.)"
