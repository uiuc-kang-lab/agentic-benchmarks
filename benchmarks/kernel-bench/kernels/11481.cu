#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
        idx += blockDim.x * gridDim.x;
    }
    
    // Warp-level reduction using shuffle intrinsics
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    
    // Write reduced value of each warp to shared memory
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        partial_sums[threadIdx.x / warpSize] = sum;
    }
    __syncthreads();

    // Final reduction of each warp's result
    int lane = threadIdx.x;
    int numWarps = blockDim.x / warpSize;
    if (lane < numWarps) {
        sum = partial_sums[lane];
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (lane == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
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