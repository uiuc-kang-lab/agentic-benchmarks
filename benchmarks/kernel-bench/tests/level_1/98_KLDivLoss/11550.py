
import os
import pytest
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F

# Utility to compile the CUDA kernel from the file "kernel.cu"
def build_kernel(kernel_filename="kernel.cu", extra_cuda_cflags=None):
    extra_cuda_cflags = extra_cuda_cflags or ["-O3", "--use_fast_math"]
    return load(
        name="kl_div_module",
        sources=[kernel_filename],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )

# Test 1: Incorrect KL divergence computation.
# For inputs where predictions and targets are identical uniform distributions,
# torch.nn.functional.kl_div returns 0, but the kernel will compute a nonzero value.
def test_incorrect_kl_div_computation(tmp_path):
    # Create a simple uniform distribution along the last dim.
    batch_size, num_classes = 16, 10
    # Uniform distribution => each probability is 1/num_classes.
    preds = torch.full((batch_size, num_classes), 1.0/num_classes, device="cuda", dtype=torch.float32)
    targets = torch.full((batch_size, num_classes), 1.0/num_classes, device="cuda", dtype=torch.float32)
    # Compute reference using PyTorch function.
    ref = F.kl_div(torch.log(preds), targets, reduction='batchmean')
    # Build the kernel module.
    module = build_kernel()
    # Our kernel expects float pointers from a flattened 1D tensor.
    # Flatten inputs.
    log_preds_flat = torch.log(preds).contiguous()
    targets_flat = targets.contiguous()
    out = module.forward(log_preds_flat, targets_flat)
    torch.cuda.synchronize()
    # For correct implementation, the divergence should be 0.
    # We expect a nonzero result due to the incorrect formula.
    assert not torch.allclose(out, torch.tensor([0.0], device="cuda"), atol=1e-5), \
        f"Kernel unexpectedly returned 0 divergence even though the formula is incorrect. Got {out.item()} vs expected 0."

# Test 2: Incorrect scaling (dividing by n rather than batch size).
# For a nontrivial distribution, compare the kernel output with the reference computed manually.
def test_incorrect_scaling(tmp_path):
    batch_size, num_classes = 8, 20
    preds = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    # Reference: computed using torch.nn.functional.kl_div takes a batchmean reduction
    ref = F.kl_div(torch.log(preds), targets, reduction='batchmean')
    module = build_kernel()
    log_preds_flat = torch.log(preds).contiguous()
    targets_flat = targets.contiguous()
    out = module.forward(log_preds_flat, targets_flat)
    torch.cuda.synchronize()
    # The kernel divides by the total number of elements (batch_size * num_classes)
    # while torch.kl_div (with reduction="batchmean") divides by batch_size.
    # They will differ by a factor of num_classes.
    expected = ref * num_classes
    # Because of the incorrect scaling, the kernel output should not match the reference.
    assert not torch.allclose(out, expected, atol=1e-5), \
        f"Kernel scaling appears correct, but it should be dividing by n instead of batch size. Kernel output: {out.item()}, expected (scaled): {expected.item()}."

# Test 3: Shared memory bounds issue (block size too large).
# We simulate this by modifying the kernel code to use a block size >1024.
# We'll create a temporary CUDA file with a modified threads_per_block.
def test_shared_memory_overflow(tmp_path):
    orig_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void balanced_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int wid = tid >> 5;  // Warp ID
    const int lid = tid & 31;  // Lane ID
    const int warps_per_block = blockDim.x >> 5;
    
    // Calculate work per thread
    const int total_threads = gridDim.x * blockDim.x;
    const int elements_per_thread = (n + total_threads - 1) / total_threads;
    const int thread_start = (blockIdx.x * blockDim.x + tid) * elements_per_thread;
    const int thread_end = thread_start + elements_per_thread > n ? n : thread_start + elements_per_thread;
    
    // Shared memory for partial sums - only need one float per warp
    __shared__ float warp_sums[32];  // Support up to 32 warps
    
    float sum = 0.0f;
    #pragma unroll 4
    for (int idx = thread_start; idx < thread_end; idx++) {
        sum += compute_kl_div(log_predictions[idx], targets[idx]);
    }
    
    sum = warp_reduce_sum(sum);
    
    if (lid == 0) {
        warp_sums[wid] = sum;
    }
    __syncthreads();
    
    if (wid == 0) {
        float warp_sum = 0.0f;
        if (lid < warps_per_block) {
            warp_sum = warp_sums[lid];
        }
        warp_sum = warp_reduce_sum(warp_sum);
        if (lid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Set an excessive threads per block to trigger shared memory overflow.
    const int threads_per_block = 2048;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    balanced_kl_div_kernel<<<num_blocks, threads_per_block, 0>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}
"""
    kernel_file = tmp_path / "kernel_overflow.cu"
    kernel_file.write_text(orig_kernel)
    
    # Build the module with the modified kernel.
    module = load(
        name="kl_div_overflow",
        sources=[str(kernel_file)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    batch_size, num_classes = 4, 64
    preds = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    
    log_preds_flat = torch.log(preds).contiguous()
    targets_flat = targets.contiguous()
    
    # The shared memory for warp_sums is only sized for 32 warps.
    # With 2048 threads per block, there are 2048/32 = 64 warps.
    # We expect the kernel to produce an incorrect result or trigger a CUDA error.
    with pytest.raises(Exception):
        out = module.forward(log_preds_flat, targets_flat)
        torch.cuda.synchronize()

# Test 4: Loop unroll issue.
# Although the pragma unroll is specified, the loop does not increment by 4.
# We simulate a situation with many elements per thread where the unroll would matter.
# Since the kernel’s result is wrong overall, we test that the accumulated error becomes larger for greater work per thread.
def test_loop_unroll_effect(tmp_path):
    batch_size, num_classes = 2, 4096  # Many elements per thread.
    preds = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    module = build_kernel()
    log_preds_flat = torch.log(preds).contiguous()
    targets_flat = targets.contiguous()
    out_small = module.forward(log_preds_flat, targets_flat)
    # Increase work per thread by replicating the input.
    preds_large = preds.repeat(1, 4)
    targets_large = targets.repeat(1, 4)
    log_preds_large = torch.log(preds_large).contiguous()
    targets_large = targets_large.contiguous()
    out_large = module.forward(log_preds_large, targets_large)
    torch.cuda.synchronize()
    # Although both computations are wrong in the same way,
    # we expect that the error (if accumulated wrongly due to loop mis-unrolling)
    # will make the ratio of the outputs different from 1.
    ratio = (out_large / out_small).item()
    assert abs(ratio - 4.0) > 1e-3, \
        f"Kernel loop unrolling appears to have no effect on error accumulation. Ratio: {ratio} (expected not close to 4)."

# Test 5: Data type check.
# Passing inputs of type float64 should cause the kernel (which expects float32) to misbehave.
def test_dtype_check(tmp_path):
    batch_size, num_classes = 8, 32
    preds = torch.softmax(torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float64), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float64), dim=-1)
    module = build_kernel()
    log_preds_flat = torch.log(preds).contiguous()
    targets_flat = targets.contiguous()
    with pytest.raises(RuntimeError):
        _ = module.forward(log_preds_flat, targets_flat)
        torch.cuda.synchronize()

# Test 6: Memory layout check.
# Passing non-contiguous tensors should trigger issues.
def test_non_contiguous_input(tmp_path):
    batch_size, num_classes = 8, 32
    preds = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    # Make the tensors non-contiguous by transposing twice.
    preds_noncontig = preds.t().t()
    targets_noncontig = targets.t().t()
    # Verify that the tensors are non-contiguous.
    assert not preds_noncontig.is_contiguous()
    module = build_kernel()
    log_preds_flat = torch.log(preds_noncontig)
    targets_flat = targets_noncontig
    with pytest.raises(RuntimeError):
        _ = module.forward(log_preds_flat, targets_flat)
        torch.cuda.synchronize()

# Test 7: Lack of CUDA error checking.
# Passing mismatched input sizes should eventually trigger an error inside the kernel.
def test_mismatched_input_sizes(tmp_path):
    batch_size, num_classes = 8, 32
    preds = torch.softmax(torch.randn(batch_size, num_classes, device="cuda"), dim=-1)
    # Create targets with a different shape.
    targets = torch.softmax(torch.randn(batch_size, num_classes + 1, device="cuda"), dim=-1)
    module = build_kernel()
    log_preds_flat = torch.log(preds).contiguous()
    targets_flat = targets.contiguous()
    with pytest.raises(RuntimeError):
        _ = module.forward(log_preds_flat, targets_flat)
        torch.cuda.synchronize()
