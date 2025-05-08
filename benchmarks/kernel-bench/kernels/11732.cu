#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel_uniform(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float shared_sums[];
    float sum = 0.0f;

    const int warp_size = 32;

    // Vector processing
    const int lanes_per_vec = 4;
    const int n_vec = n / lanes_per_vec;
    const int n_rem = n % lanes_per_vec;

    const float4* vec_predictions = reinterpret_cast<const float4*>(log_predictions);
    const float4* vec_targets = reinterpret_cast<const float4*>(targets);

    // Process entire float4 vectors
    for (int i = idx; i < n_vec; i += blockDim.x * gridDim.x) {
        float4 pred = vec_predictions[i];
        float4 targ = vec_targets[i];

        sum += expf(pred.x) - targ.x * pred.x +
               expf(pred.y) - targ.y * pred.y +
               expf(pred.z) - targ.z * pred.z +
               expf(pred.w) - targ.w * pred.w;
    }


    // Process remaining elements up to block boundary
    int rem_start = n_vec * lanes_per_vec;
    for (int i = rem_start + idx; i < n; i += blockDim.x * gridDim.x) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }

    shared_sums[threadIdx.x] = sum;
    __syncthreads();

    // Perform block-wide reduction in shared memory
    for (int stride = warp_size / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x % warp_size < stride) {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Only the threads at the beginning of the warp will proceed
    if (threadIdx.x % warp_size == 0) {
        atomicAdd(output, shared_sums[threadIdx.x]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel_uniform<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA Uniform)");
}