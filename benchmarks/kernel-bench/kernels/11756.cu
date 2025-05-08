#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void warp_reduced_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    constexpr int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const unsigned int mask = 0xffffffff;
    
    extern __shared__ float warp_sums[];
    float thread_sum = 0.0f;

    // Strided loop for coalesced memory access
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
    }

    // Warp-level reduction
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Store warp sums in shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp reduces all warp sums
    if (warp_id == 0) {
        float warp_total = (lane_id < blockDim.x/warp_size) ? warp_sums[lane_id] : 0.0f;
        
        for (int offset = blockDim.x/(warp_size*2); offset > 0; offset >>= 1) {
            warp_total += __shfl_down_sync(mask, warp_total, offset);
        }

        if (lane_id == 0) {
            atomicAdd(output, warp_total);
        }
    }
}

torch::Tensor warp_reduced_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Optimized launch config
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks = std::min(512, (n + threads - 1) / threads);
    const int shared_mem = warps_per_block * sizeof(float);

    warp_reduced_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_reduced_kl_forward, "Warp-reduced KL divergence (CUDA)");
}
