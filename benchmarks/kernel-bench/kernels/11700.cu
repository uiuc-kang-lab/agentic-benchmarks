#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__device__ __inline__ float warp_reduce(float sum) {
    const int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

template<int BLOCK_SIZE>
__device__ __inline__ float block_reduce(float sum, float* shared) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    sum = warp_reduce<BLOCK_SIZE>(sum);
    
    if (lane_id == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < BLOCK_SIZE/32) ? shared[lane_id] : 0.0f;
        sum = warp_reduce<BLOCK_SIZE/32>(sum);
    }
    return sum;
}

template<int BLOCK_SIZE>
__global__ void kl_div_modular_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int n) {
    
    __shared__ float shared[BLOCK_SIZE/32];
    float sum = 0.0f;
    
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; 
         i < n; 
         i += BLOCK_SIZE * gridDim.x) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += expf(log_pred) - target * log_pred;
    }

    sum = block_reduce<BLOCK_SIZE>(sum, shared);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor modular_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    constexpr int BLOCK_SIZE = 256;
    const int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int NUM_WARPS = BLOCK_SIZE / 32;
    
    kl_div_modular_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, NUM_WARPS*sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modular_kl_div_forward, "Modular KLDivLoss with optimized reductions (CUDA)");
}
