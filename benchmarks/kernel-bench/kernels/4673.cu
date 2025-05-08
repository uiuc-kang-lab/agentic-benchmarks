#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rms_norm_kernel_shared(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    extern __shared__ float shared_mem[];
    float* shared_sums = shared_mem;
    float* shared_rms = &shared_mem[blockDim.x];

    const int batch_element_id = blockIdx.x;
    const int batch_id = batch_element_id / numel_per_batch;
    const int offset_in_batch = batch_element_id % numel_per_batch;
    
    if (batch_id >= batch_size) return;

    const int base_offset = batch_id * num_features * numel_per_batch + offset_in_batch;
    const int tid = threadIdx.x;

    // Phase 1: Calculate sum of squares
    float local_sum = 0.0f;
    const int features_per_thread = (num_features + blockDim.x - 1) / blockDim.x;
    
    for (int i = 0; i < features_per_thread; i++) {
        const int feature_idx = tid + i * blockDim.x;
        if (feature_idx < num_features) {
            const scalar_t val = input[base_offset + feature_idx * numel_per_batch];
            local_sum += static_cast<float>(val) * static_cast<float>(val);
        }
    }

    shared_sums[tid] = local_sum;
    __syncthreads();

    // Block-wide reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_sums[tid] += shared_sums[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float mean_sq = shared_sums[0] / num_features;
        shared_rms[0] = sqrtf(mean_sq + eps);
    }
    __syncthreads();

    // Phase 2: Normalize with shared RMS
    const scalar_t rms = static_cast<scalar_t>(shared_rms[0]);
    
    for (int i = 0; i < features_per_thread; i++) {
        const int feature_idx = tid + i * blockDim.x;
        if (feature_idx < num_features) {
            const int idx = base_offset + feature_idx * numel_per_batch;
            output[idx] = input[idx] / rms;
        }
    }
}

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); ++i) {
        numel_per_batch *= input.size(i);
    }

    const int total_blocks = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const size_t shared_mem_size = (threads_per_block + 1) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel_shared<scalar_t><<<total_blocks, threads_per_block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            numel_per_batch,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA)");
}
