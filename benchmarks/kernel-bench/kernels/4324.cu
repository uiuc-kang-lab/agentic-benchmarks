#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// This kernel uses a two-level loop design to ensure memory coalescing:
// It partitions the (N * H) rows among warps so that within each row, threads read consecutive columns.
// The reduction phase uses warp-level shuffles and shared memory to compute the channel-wise mean and variance.

__global__ void batch_norm_coalesced_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    bool training,
    float momentum,
    float eps,
    float* __restrict__ output,
    int N,
    int C,
    int H,
    int W) {

    // Each block handles one channel
    int c = blockIdx.x;
    if (c >= C) return;

    // Total number of rows per channel (each row is one instance of H dimension for a given N)
    int num_rows = N * H;  // each row has W elements
    int total_elements = num_rows * W;  // total elements in this channel

    // Determine warp information
    const int warpSize = 32;
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    int num_warps = blockDim.x / warpSize; // assuming blockDim.x is a multiple of warpSize

    // Each thread will accumulate a partial sum and sum of squares over elements it reads
    float local_sum = 0.f;
    float local_sum_sq = 0.f;

    // Loop over rows assigned to this warp. There are num_rows rows, each with W contiguous elements.
    // Each warp processes rows in a strided manner based on its warp_id.
    for (int row = warp_id; row < num_rows; row += num_warps) {
        // Compute the corresponding n and h indices
        int n = row / H;        // image index
        int h = row % H;        // row within the image
        // Base address for this (n, c, h, :) row
        int base = n * C * H * W + c * H * W + h * W;

        // Each lane processes a subset of the W columns; this guarantees coalesced accesses
        for (int w = lane; w < W; w += warpSize) {
            int idx = base + w;
            float val = input[idx];
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    // Warp-level reduction using shuffle instructions
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum    += __shfl_down_sync(mask, local_sum, offset);
        local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
    }

    // Use shared memory to accumulate results across warps
    extern __shared__ float shared_mem[]; // size: 2 * num_warps * sizeof(float)
    // First half for warp sums, second half for warp sum-of-squares
    float* warp_sums = shared_mem;
    float* warp_sum_sq_arr = &shared_mem[num_warps];

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
        warp_sum_sq_arr[warp_id] = local_sum_sq;
    }
    __syncthreads();

    float mean = 0.f;
    float var = 0.f;
    
    // Thread 0 aggregates the results from all warps
    if (tid == 0) {
        float total_sum = 0.f;
        float total_sum_sq = 0.f;
        for (int i = 0; i < num_warps; i++) {
            total_sum += warp_sums[i];
            total_sum_sq += warp_sum_sq_arr[i];
        }
        mean = total_sum / total_elements;
        var = total_sum_sq / total_elements - mean * mean;
        
        // Update running stats if training
        if (training) {
            running_mean[c] = (1.f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.f - momentum) * running_var[c] + momentum * var;
        }
        // Store mean and variance in shared memory for reuse by all threads
        warp_sums[0] = mean;
        warp_sums[1] = var;
    }
    __syncthreads();

    // All threads retrieve the computed mean and variance
    mean = warp_sums[0];
    var = warp_sums[1];
    float inv_std = 1.0f / sqrtf(var + eps);

    // Read weight and bias for this channel
    float w_val = weight[c];
    float b_val = bias[c];

    // Normalization: loop over the same rows to compute the normalized output
    for (int row = warp_id; row < num_rows; row += num_warps) {
        int n = row / H;
        int h = row % H;
        int base = n * C * H * W + c * H * W + h * W;
        
        for (int col = lane; col < W; col += warpSize) {
            int idx = base + col;
            float val = input[idx];
            output[idx] = (val - mean) * inv_std * w_val + b_val;
        }
    }
}


// Kernel launcher for the coalesced BatchNorm
torch::Tensor forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {

    // Checks
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    CHECK_CUDA(running_mean);
    CHECK_CUDA(running_var);

    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(running_mean);
    CHECK_CONTIGUOUS(running_var);

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);

    const int threads = 256;  // should be a multiple of warpSize (32)
    int num_warps = threads / 32;
    // Shared memory: 2 arrays of floats of size (num_warps)
    size_t shared_mem_bytes = 2 * num_warps * sizeof(float);

    // Launch one block per channel
    batch_norm_coalesced_kernel<<<C, threads, shared_mem_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        training,
        momentum,
        eps,
        output.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Coalesced BatchNorm forward (CUDA)");
}
