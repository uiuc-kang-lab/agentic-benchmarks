
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1:
# This test sends float32 inputs and compares the result from the CUDA kernel
# with the expected output from PyTorch's native F.kl_div.
# Due to the incorrect formula, the two results should differ.
def test_incorrect_formula_and_reduction():
    torch.manual_seed(0)
    batch_size = 128
    input_shape = (4096,)
    predictions = torch.randn(batch_size, *input_shape, device="cuda").softmax(dim=-1)
    # Create targets as a probability distribution
    targets = torch.randn(batch_size, *input_shape, device="cuda").softmax(dim=-1)
    # Compute reference using F.kl_div with log(predictions) as input.
    ref = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    
    my_module = build_kernel()
    # Launch the CUDA kernel using float32 tensors.
    out = my_module.forward(torch.log(predictions), targets)
    # Due to wrong formula and wrong reduction normalization,
    # the outputs should not be close.
    assert not torch.allclose(out, ref, atol=1e-4), \
        f"Expected different result due to incorrect KL divergence implementation, got {out} vs {ref}"

# Test case 2:
# This test sends double precision inputs to check that the kernel fails or produces incorrect results
# because it only supports float32.
def test_input_tensor_type():
    torch.manual_seed(0)
    batch_size = 32
    input_shape = (1024,)
    # Create double precision tensors
    predictions = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.double).softmax(dim=-1)
    targets = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.double).softmax(dim=-1)
    
    my_module = build_kernel()
    # Expecting an error or wrong behavior because kernel uses float pointers.
    with pytest.raises(RuntimeError):
        # The kernel expects float32. Conversion might be needed,
        # so here's a deliberate wrong usage.
        _ = my_module.forward(torch.log(predictions), targets)

# Test case 3:
# This test is designed to highlight the wrong reduction normalization.
# For a small batch (e.g. 1 sample) the batchmean should equal the sum (since batch size is 1),
# but the kernel divides by the total number of elements.
def test_wrong_reduction_normalization():
    torch.manual_seed(0)
    batch_size = 1  # single sample to see effect of reduction factor
    input_shape = (2048,)
    predictions = torch.randn(batch_size, *input_shape, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch_size, *input_shape, device="cuda").softmax(dim=-1)
    
    ref = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    my_module = build_kernel()
    out = my_module.forward(torch.log(predictions), targets)
    
    # When batch size is 1, the correct reduction should yield the sum of losses,
    # but our kernel divides by n, so the result will be scaled down.
    assert not torch.allclose(out, ref, atol=1e-5), \
        f"Expected different result due to wrong normalization, got {out} vs {ref}"
        
if __name__ == "__main__":
    pytest.main([__file__])
