#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    extern __shared__ float partial_sums[];
    float sum = 0.0f;
    
    // Process 4 elements per thread with grid-stride loop
    const int tid = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    for(int i = 0; i < 4; ++i) {
        int idx = tid + i * blockDim.x;
        if(idx < n) {
            float log_pred = log_predictions[idx];
            float target = targets[idx];
            sum += target * (log_pred - logf(target));
        }
    }
    
    // Shared memory reduction
    partial_sums[threadIdx.x] = sum;
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Final warp-level reduction with shuffle
    if(threadIdx.x < 32) {
        float val = partial_sums[threadIdx.x];
        for(int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        
        if(threadIdx.x == 0)
            atomicAdd(output, val);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    int sm_count;
cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    const int blocks_per_sm = 8;
    int blocks = min((n + threads*4 - 1) / (threads*4), sm_count * blocks_per_sm);
    
    kl_div_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward optimized (CUDA)");
}