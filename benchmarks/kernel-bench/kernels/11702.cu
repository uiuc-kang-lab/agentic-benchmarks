#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void warp_convergent_kl_div_kernel(
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
    
    // Uniform memory access pattern for entire warp
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += __expf(log_pred) - target * log_pred;
    }

    // Warp-level reduction without branching
    for(int offset = warp_size/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Warp leaders store to shared memory
    if(lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // First warp handles final reduction
    if(warp_id == 0) {
        float val = lane_id < warps_per_block ? warp_sums[lane_id] : 0.0f;
        
        for(int offset = warp_size/2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if(lane_id == 0) {
            atomicAdd(output, val);
        }
    }
}

torch::Tensor warp_convergent_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Auto-tuned launch configuration
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    const int shared_mem = (block_size / 32) * sizeof(float);

    warp_convergent_kl_div_kernel<<<
        min(512, grid_size), 
        block_size, 
        shared_mem>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_convergent_kl_div_forward, "Warp-convergent KLD (CUDA)");
}
