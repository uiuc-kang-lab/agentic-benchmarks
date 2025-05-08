#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const unsigned int warp_size = 32;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    const unsigned int warps_per_block = blockDim.x / warp_size;
    
    extern __shared__ float warp_results[];
    
    float sum = 0.0f;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n; i += stride) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_sum = (lane_id < warps_per_block) ? warp_results[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

void kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets,
    torch::Tensor output,
    cudaStream_t stream) {
    
    const int n = log_predictions.numel();
    
    const int threads_per_block = 256;
    const int num_warps = threads_per_block / 32;
    const int blocks = min(256, (n + threads_per_block - 1) / threads_per_block);
    const int shared_mem = num_warps * sizeof(float);
    
    kl_div_kernel<<<blocks, threads_per_block, shared_mem, stream>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", [](torch::Tensor log_predictions, torch::Tensor targets) {
        auto output = torch::zeros({1}, log_predictions.options());
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        kl_div_cuda_forward(log_predictions, targets, output, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return output / static_cast<float>(log_predictions.numel());
    }, "KL divergence forward (CUDA)");
}