#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void unrolled_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    
    extern __shared__ float warp_sums[];
    
    float sum = 0.0f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Unrolled main computation loop
    #pragma unroll 4
    for (int i = tid; i < n; i += stride) {
        const float log_pred = __ldg(&log_predictions[i]);
        const float target = __ldg(&targets[i]);
        sum += expf(log_pred) - target * log_pred;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final block reduction
    if (warp_id == 0 && lane_id < warps_per_block) {
        float val = warp_sums[lane_id];
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        if (lane_id == 0) {
            atomicAdd(output, val);
        }
    }
}

torch::Tensor unrolled_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Block size heuristics
    int block_size = (n > 1 << 18) ? 512 : 
                    (n < 1 << 14) ? 128 : 256;
    
    const int max_blocks = 256;
    int blocks = min(max_blocks, (n + block_size - 1) / block_size);
    const int shared_mem = (block_size / warp_size) * sizeof(float);

    unrolled_kl_div_kernel<<<blocks, block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &unrolled_kl_div_forward, "KLDivLoss with unrolled loops (CUDA)");
}