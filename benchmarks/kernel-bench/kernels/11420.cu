#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// First kernel: Each block computes its partial sum without global atomics
__global__ void partial_reduce_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ partial,
    const int n) {

    extern __shared__ float sdata[];  // Shared memory for intra-block reduction
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop to compute local sum
    for (; idx < n; idx += stride) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduce within the block in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block's result to the partial sums array
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// Second kernel: Final reduction of the partial block sums
__global__ void final_reduce_kl_div_kernel(
    const float* __restrict__ partial,
    float* __restrict__ output,
    const int partial_count) {

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    // Each thread sums over multiple elements of the partial array
    for (int i = tid; i < partial_count; i += blockDim.x) {
        sum += partial[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Standard reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write final result
    if (tid == 0) {
        output[0] = sdata[0];
    }
}

// Host function launching two kernels: one for partial reduction and one for final reduction
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    const int threads = 256;
    int computed_blocks = (n + threads - 1) / threads;
    // Limit blocks to 256 to avoid launching excessive blocks for the final reduction
    const int blocks = (computed_blocks < 256) ? computed_blocks : 256;

    // Allocate temporary tensor for partial results (one per block)
    auto partial_sums = torch::empty({blocks}, log_predictions.options());

    // Launch first kernel: each block reduces its portion of the input
    partial_reduce_kl_div_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        n
    );

    // Allocate output tensor to hold the final result
    auto output = torch::zeros({1}, log_predictions.options());

    // Launch second kernel: final reduction of partial sums using a single block
    const int final_threads = 256;
    const int final_blocks = 1;
    final_reduce_kl_div_kernel<<<final_blocks, final_threads, final_threads * sizeof(float)>>>(
        partial_sums.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks
    );

    // Return the average result
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}
