
import os
import tempfile
import pytest
import torch
from torch.utils.cpp_extension import load

# Function to build the kernel from the specified source file.
def build_kernel(source_file="kernel.cu", extra_cuda_cflags=None):
    cuda_module = load(
        name="test_module",
        sources=[source_file],
        extra_cuda_cflags=extra_cuda_cflags if extra_cuda_cflags else ["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Invalid data type (issue 1)
def test_invalid_dtype():
    my_module = build_kernel()
    # Create tensors of type double
    predictions = torch.randn(128, 4096, device='cuda', dtype=torch.double)
    targets = torch.randn(128, 4096, device='cuda', dtype=torch.double)
    # Since the kernel interprets memory as float, the result is undefined.
    # We expect that the returned result is not equal to PyTorch's smooth_l1_loss.
    output_kernel = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    output_ref = torch.nn.functional.smooth_l1_loss(predictions.float(), targets.float())
    # Because the kernel read the double bits as float,
    # the result will be very different from the reference.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-3), (
        f"Kernel unexpectedly produced a correct result when given double tensors."
    )

# Test 2: Large block size causing shared memory overflow (issue 2)
def test_large_block_size():
    # We override block_size by modifying the kernel source.
    # We create a temporary kernel file with a modified block size.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False) as tmp:
        kernel_code = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#ifndef CUSTOM_BLOCK_SIZE
#define CUSTOM_BLOCK_SIZE 256
#else
#define CUSTOM_BLOCK_SIZE CUSTOM_BLOCK_SIZE
#endif

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float thread_sum = 0.0f;
    for (int i = idx; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
    }

    // Use efficient warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Each warp's leader writes its sum to shared memory.
    // WARNING: This shared memory buffer is hardcoded to size 32.
    __shared__ float shared_sums[32];
    int lane = threadIdx.x % warpSize;
    int warp_idx = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sums[warp_idx] = thread_sum;
    }
    __syncthreads();

    // Thread 0 aggregates the sums from each warp and performs an atomic add.
    if (warp_idx == 0 && lane == 0) {
        float block_sum = 0.0f;
        // Using blockDim.x / warpSize might exceed 32 if CUSTOM_BLOCK_SIZE is large.
        for (int i = 0; i < blockDim.x / warpSize; i++) {
            block_sum += shared_sums[i];
        }
        atomicAdd(output, block_sum / n_elements);
    }
}

torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = CUSTOM_BLOCK_SIZE;
    const int grid_size = std::min((n + block_size - 1) / block_size, 65535);
    smooth_l1_loss_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss (CUDA)");
}
'''
        # Set a custom block size greater than 1024 (e.g., 1056 produces 33 warps)
        tmp.write(kernel_code)
        tmp_filename = tmp.name

    # Now compile the kernel with CUSTOM_BLOCK_SIZE defined as 1056.
    try:
        my_module = load(
            name="test_module_custom_block",
            sources=[tmp_filename],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-DCUSTOM_BLOCK_SIZE=1056"],
            with_cuda=True,
            verbose=False,
        )
        # Create a tensor with exactly 1056 elements so that one block is launched.
        n_elements = 1056
        predictions = torch.randn(n_elements, device="cuda", dtype=torch.float32)
        targets = torch.randn(n_elements, device="cuda", dtype=torch.float32)
        output_kernel = my_module.forward(predictions, targets)
        torch.cuda.synchronize()
        output_ref = torch.nn.functional.smooth_l1_loss(predictions, targets)
        # The overflow in shared memory indexing should corrupt the result.
        # We expect that the kernel output deviates significantly from the reference.
        assert not torch.allclose(output_kernel, output_ref, atol=1e-4), (
            f"Kernel output should be corrupted due to shared memory overflow with large block size."
        )
    finally:
        os.remove(tmp_filename)

# Test 3: Lack of kernel error checking (issue 3)
def test_no_error_checking():
    my_module = build_kernel()
    # Force an error by providing non-contiguous tensors.
    predictions = torch.randn(128, 4096, device="cuda", dtype=torch.float32).t()  # Transposition makes it non-contiguous.
    targets = predictions.clone()
    with pytest.raises(RuntimeError, match="Input tensors must be contiguous"):
        my_module.forward(predictions, targets)

# Test 4: Handling of empty tensor input (issue 4)
def test_empty_input():
    my_module = build_kernel()
    predictions = torch.tensor([], device="cuda", dtype=torch.float32)
    targets = torch.tensor([], device="cuda", dtype=torch.float32)
    # Expect the kernel to throw an error (likely due to division by zero) or produce NaN.
    output = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    # The resulting output should be NaN or raise an error.
    assert torch.isnan(output).all(), "Output for empty input should be NaN due to division by zero."

# Test 5: Use of non-namespaced min (issue 5)
def test_min_usage():
    # We simulate a large input to force computation of grid_size.
    # This test does not trigger a runtime error in the kernel (since the module compiled) but
    # ensures that the grid_size computation does what is expected.
    my_module = build_kernel()
    n_elements = 10**8  # artificially large number of elements
    predictions = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    targets = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    output_kernel = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    output_ref = torch.nn.functional.smooth_l1_loss(predictions, targets)
    # The result should be close in value (the math is correct) even though the min usage might be suspect.
    assert torch.allclose(output_kernel, output_ref, atol=1e-3), (
        "Kernel output differs from reference; check grid_size computation using min"
    )
