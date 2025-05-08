#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel with block size experimentation
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;

    // Vector processing using float4 for 128-bit aligned accesses
    const int n4 = n / 4;
    const float4* logp_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets);

    // Process vector elements using __ldg for read-only cache
    int vec_idx = idx;
    while (vec_idx < n4) {
        float4 logp = __ldg(&logp_vec[vec_idx]);
        float4 targ = __ldg(&targ_vec[vec_idx]);
        sum += expf(logp.x) - targ.x * logp.x
             + expf(logp.y) - targ.y * logp.y
             + expf(logp.z) - targ.z * logp.z
             + expf(logp.w) - targ.w * logp.w;
        vec_idx += gridDim.x * blockDim.x;
    }

    // Process remaining elements using scalar __ldg
    int scalar_idx = n4 * 4 + idx;
    while (scalar_idx < n) {
        float log_pred = __ldg(&log_predictions[scalar_idx]);
        float target = __ldg(&targets[scalar_idx]);
        sum += expf(log_pred) - target * log_pred;
        scalar_idx += gridDim.x * blockDim.x;
    }

    // Reduction in shared memory
    partial_sums[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

// Host function to launch the kernel with different block sizes

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Experiment with different block sizes
    const int block_sizes[] = {32, 64, 128, 256, 512};
    int optimal_block_size = 256; // Default block size
    float min_time = FLT_MAX;

    for (int i = 0; i < 5; ++i) {
        const int threads = block_sizes[i];
        const int blocks = min((n / 4 + threads - 1) / threads, 1024);
        const int shared_mem = threads * sizeof(float);

        // Measure execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        kl_div_kernel<<<blocks, threads, shared_mem>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (milliseconds < min_time) {
            min_time = milliseconds;
            optimal_block_size = threads;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Launch kernel with optimal block size
    const int threads = optimal_block_size;
    const int blocks = min((n / 4 + threads - 1) / threads, 1024);
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
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA Optimized with Block Size Experimentation)");
}
