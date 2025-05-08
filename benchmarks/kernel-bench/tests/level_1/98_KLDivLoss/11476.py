
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Test 1: Race condition test.
# Running the kernel multiple times with a large input may yield inconsistent results.
def test_race_condition():
    my_module = build_kernel()
    # Create an input that forces many blocks: ensure n >> threads per block.
    batch = 256
    input_size = 1 << 16  # large input; total element count will require many blocks.
    # Create predictions and targets with softmax normalization along last dimension.
    preds = torch.randn(batch, input_size, device='cuda')
    preds = preds.softmax(dim=-1)
    targets = torch.randn(batch, input_size, device='cuda')
    targets = targets.softmax(dim=-1)
    # Compute kernel output multiple times.
    outputs = []
    for _ in range(10):
        out = my_module.forward(torch.log(preds), targets)
        torch.cuda.synchronize()
        outputs.append(out.item())
    # If there is a race condition results may vary. We check that not all outputs are almost equal.
    if all(abs(outputs[i] - outputs[0]) < 1e-5 for i in range(len(outputs))):
        pytest.skip("Race condition not triggered in this test run (may depend on scheduling), "
                    "but the potential issue remains present.")
    else:
        # If differences found, we assert they are significant (which is an indication of a race issue).
        diff = max(outputs) - min(outputs)
        assert diff > 1e-3, f"Inconsistent results detected (diff {diff}) likely due to race condition in final reduction."

# Test 2: Normalization factor test.
# Provide a simple input and compare with expected normalization (batchmean vs total number of elements).
def test_normalization():
    my_module = build_kernel()
    batch = 4
    num_features = 10
    # Create predictions and targets; to have a predictable KL divergence, set target = prediction.
    # In torch.nn.functional.kl_div with reduction='batchmean', if target==prediction then KL=0.
    preds = torch.full((batch, num_features), 0.1, dtype=torch.float32, device='cuda')
    preds = preds.softmax(dim=-1)
    targets = preds.clone()
    # Torch KL divergence: when predictions exactly equal targets, the loss should be zero.
    loss_torch = torch.nn.functional.kl_div(torch.log(preds), targets, reduction='batchmean')
    loss_cuda = my_module.forward(torch.log(preds), targets)
    torch.cuda.synchronize()
    # Kernel divides by total number of elements; unless batch==num_elements this will not match.
    assert not torch.allclose(loss_cuda, loss_torch, atol=1e-5), \
            f"Normalization issue not detected! Torch kl_div: {loss_torch.item()}, Kernel: {loss_cuda.item()}"

# Test 3: Incorrect KL divergence formula.
# Using known inputs where the expected formula produces a known result.
def test_incorrect_formula():
    my_module = build_kernel()
    batch = 2
    num_features = 5
    # Create inputs with fixed values.
    # Let predictions be such that log(predictions) = log(0.2) constant, and targets be constant 0.2.
    pred_val = 0.2
    log_pred_val = torch.log(torch.tensor(pred_val))
    preds = torch.full((batch, num_features), pred_val, dtype=torch.float32, device='cuda')
    targets = torch.full((batch, num_features), pred_val, dtype=torch.float32, device='cuda')
    preds = preds.softmax(dim=-1)
    targets = targets.softmax(dim=-1)
    
    # Compute expected output using torch.nn.functional.kl_div:
    loss_torch = torch.nn.functional.kl_div(torch.log(preds), targets, reduction='batchmean')
    loss_cuda = my_module.forward(torch.log(preds), targets)
    torch.cuda.synchronize()
    # They should be equal if the kernel formula were correct.
    assert not torch.allclose(loss_cuda, loss_torch, atol=1e-5), \
           f"Formula issue not detected! Torch kl_div: {loss_torch.item()}, Kernel: {loss_cuda.item()}"

# Test 4: Data type incompatibility.
# Provide double type tensors and expect the kernel to either fail or return wrong results.
def test_tensor_dtype():
    my_module = build_kernel()
    batch = 2
    num_features = 100
    preds = torch.randn(batch, num_features, dtype=torch.float64, device='cuda')
    preds = preds.softmax(dim=-1)
    targets = torch.randn(batch, num_features, dtype=torch.float64, device='cuda')
    targets = targets.softmax(dim=-1)
    with pytest.raises(RuntimeError):
        # The kernel expects float32; passing float64 should raise an error.
        _ = my_module.forward(torch.log(preds), targets)
