#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256

__device__ __forceinline__ float compute_kl_div(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void block_reduce_sum(float* shared_data, const int tid) {
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    
    if (warp_id == 0) {
        float sum = (tid < blockDim.x/32) ? shared_data[tid] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            shared_data[0] = sum;
        }
    }
}

__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {

    extern __shared__ float shared_mem[];
    float* shared_log_pred = shared_mem;
    float* shared_targets = &shared_mem[TILE_SIZE];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int gid = bid * num_threads + tid;

    float thread_sum = 0.0f;

    // Grid stride loop
    for (int idx = gid; idx < n; idx += gridDim.x * num_threads) {
        if(idx < n) { 
            float log_pred = log_predictions[idx];
            float target = targets[idx];
            thread_sum += compute_kl_div(log_pred, target);
        }
    }

    // Warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);

    // Write warp results to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = thread_sum;
    }
    __syncthreads();

    // Block-level reduction
    if (warp_id == 0) {
        block_reduce_sum(shared_mem, tid);
        if (tid == 0) {
            atomicAdd(output, shared_mem[0]);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = TILE_SIZE;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = (threads/32) * sizeof(float);
    
    optimized_kl_div_kernel<<<blocks, threads, shared_mem>>>(
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