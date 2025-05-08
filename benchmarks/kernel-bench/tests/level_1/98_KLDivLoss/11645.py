
import torch
import pytest
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper: Compute the "correct" KL divergence as used by torch.nn.functional.kl_div 
# with log_input and reduction="batchmean". In PyTorch this is computed as:
# loss = target * (log(target) - log_prediction)
def correct_kl_div(log_predictions, targets):
    # We assume log_predictions = log(predictions) so that exp(log_predictions) = predictions.
    # and the element-wise divergence is defined as:
    # t * (log(t) - log_p)  (and then summed and divided by batch size)
    loss_elements = targets * (torch.log(targets.clamp_min(1e-10)) - log_predictions)
    # batchmean divides by batch size (first dimension); sum over rest dims.
    batch_size = log_predictions.shape[0]
    return loss_elements.sum() / batch_size

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestKLDivergenceKernel:

    def test_incorrect_formula(self):
        # Use controlled input so that the correct loss should be zero.
        # For example, if predictions == targets then KL divergence = target * (log(target) - log(target)) == 0.
        batch_size = 4
        features = 10
        # Create uniform distributions.
        predictions = torch.full((batch_size, features), 1.0/float(features), device="cuda", dtype=torch.float32)
        targets = torch.full((batch_size, features), 1.0/float(features), device="cuda", dtype=torch.float32)
        log_predictions = predictions.log()

        kernel_module = build_kernel()
        # Call the CUDA kernel extension
        output = kernel_module.forward(log_predictions, targets)
        torch.cuda.synchronize()

        # The correct KL divergence (with reduction 'batchmean') should be 0
        expected = correct_kl_div(log_predictions, targets)
        # Since our kernel uses a wrong formula, we expect a nonzero result.
        assert not math.isclose(output.item(), expected.item(), rel_tol=1e-3), \
            "Test failed to detect incorrect KL divergence formula (output unexpectedly close to expected 0.)"

    def test_incorrect_normalization(self):
        # In torch.nn.functional.kl_div with reduction="batchmean", the final loss is divided by the batch size,
        # independent of the per-sample number of elements.
        # Our kernel divides by total element count.
        batch_size = 8
        # Two different feature dimensions.
        features_small = 4
        features_large = 16

        # For simplicity, use random distributions.
        predictions_small = torch.randn(batch_size, features_small, device="cuda", dtype=torch.float32).softmax(dim=-1)
        targets_small = torch.randn(batch_size, features_small, device="cuda", dtype=torch.float32).softmax(dim=-1)
        log_predictions_small = predictions_small.log()

        predictions_large = torch.randn(batch_size, features_large, device="cuda", dtype=torch.float32).softmax(dim=-1)
        targets_large = torch.randn(batch_size, features_large, device="cuda", dtype=torch.float32).softmax(dim=-1)
        log_predictions_large = predictions_large.log()

        kernel_module = build_kernel()
        output_small = kernel_module.forward(log_predictions_small, targets_small)
        output_large = kernel_module.forward(log_predictions_large, targets_large)
        torch.cuda.synchronize()

        # Compute "correct" KL divergence (ignoring the formula error, we focus on normalization).
        expected_small = correct_kl_div(log_predictions_small, targets_small)
        expected_large = correct_kl_div(log_predictions_large, targets_large)

        # The correct losses should be roughly the same order (if distributions are random).
        # However, due to division by total element count in the kernel,
        # the kernel's computed losses will be scaled by 1/features.
        factor = features_large / features_small  # expect that kernel_large ~= kernel_small / factor
        ratio = output_small.item() / output_large.item() if output_large.item() != 0 else 0.0

        # If the kernel applied proper normalization, ratio would be near 1.
        # Here, we expect the ratio to differ significantly.
        assert abs(ratio - 1.0) > 0.1, \
            "Test failed to detect incorrect normalization: kernel output ratio is too close to 1."

    def test_warp_reduction_issue(self):
        # Force a scenario where the total number of elements is not a multiple of the typical warp size (32)
        # so that some warps have incomplete occupancy.
        batch_size = 3
        # Use a feature dimension that is not a multiple of 32.
        features = 31
        # Use random distributions.
        predictions = torch.randn(batch_size, features, device="cuda", dtype=torch.float32).softmax(dim=-1)
        targets = torch.randn(batch_size, features, device="cuda", dtype=torch.float32).softmax(dim=-1)
        log_predictions = predictions.log()

        kernel_module = build_kernel()
        output = kernel_module.forward(log_predictions, targets)
        torch.cuda.synchronize()

        # Compute the expected result using our reference KL divergence.
        expected = correct_kl_div(log_predictions, targets)
        # Even if the formula is incorrect, the warp reduction issue could further amplify error,
        # so we check that the kernel output is far from the expected value.
        diff = abs(output.item() - expected.item())
        assert diff > 1e-3, \
            f"Test failed to trigger warp reduction issue: difference ({diff}) is too small."
