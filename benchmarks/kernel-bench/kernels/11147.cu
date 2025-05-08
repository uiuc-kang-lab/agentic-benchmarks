#include <torch/extension.h>
#include <cuda_fp16.h>

__forceinline__ __device__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ float compute_max_logit(const float* __restrict__ logits, int num_classes) {
    float max_val = -INFINITY;
    int i = 0;
    
    // Vectorized max reduction
    for (; i <= num_classes - 4; i += 4) {
        float4 vals = __ldg(reinterpret_cast<const float4*>(logits + i));
        max_val = fmaxf(max_val, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
    }
    
    // Handle remaining elements
    for (; i < num_classes; i++) {
        max_val = fmaxf(max_val, __ldg(logits + i));
    }
    return max_val;
}

__device__ float compute_exp_sum(float max_val, const float* __restrict__ logits, int num_classes) {
    float sum = 0.0f;
    int i = 0;
    
    // Vectorized exp sum
    for (; i <= num_classes - 4; i += 4) {
        float4 vals = __ldg(reinterpret_cast<const float4*>(logits + i));
        sum += expf(vals.x - max_val);
        sum += expf(vals.y - max_val);
        sum += expf(vals.z - max_val);
        sum += expf(vals.w - max_val);
    }
    
    // Handle remaining elements
    for (; i < num_classes; i++) {
        sum += expf(__ldg(logits + i) - max_val);
    }
    return sum;
}

__global__ void cross_entropy_loss_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = idx; i < batch_size; i += total_threads) {
        const float* sample_logits = logits + i * num_classes;
        int target = __ldg(targets + i);
        
        float max_logit = compute_max_logit(sample_logits, num_classes);
        float sum_exp = compute_exp_sum(max_logit, sample_logits, num_classes);
        float log_sum_exp = logf(sum_exp);
        
        losses[i] = -(__ldg(sample_logits + target) - max_logit - log_sum_exp);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be CUDA tensor");
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    auto losses = torch::empty({batch_size}, predictions.options());

    // Optimized launch configuration
    const int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    blocks = min(blocks, 128);

    cross_entropy_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CE Loss with vectorized loads and read-only caching");
}
