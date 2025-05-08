#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ float parallel_sumsq(
    const scalar_t* __restrict__ input,
    const int base_offset,
    const int numel_per_batch,
    const int num_features,
    float* shared_sum
) {
    const int tid = threadIdx.x;
    const int features_per_thread = (num_features + blockDim.x - 1) / blockDim.x;
    const int start = tid * features_per_thread;
    const int end = min(start + features_per_thread, num_features);

    float local_sum = 0.0f;
    for (int feat = start; feat < end; ++feat) {
        const scalar_t val = input[base_offset + feat * numel_per_batch];
        local_sum += static_cast<float>(val) * static_cast<float>(val);
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }
    return shared_sum[0];
}

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    __shared__ float shared_mem[256];
    
    const int batch_element_id = blockIdx.x;
    const int batch_id = batch_element_id / numel_per_batch;
    const int offset_in_batch = batch_element_id % numel_per_batch;
    
    if (batch_id >= batch_size) return;

    const int base_offset = batch_id * num_features * numel_per_batch + offset_in_batch;
    const float sumsq = parallel_sumsq(input, base_offset, numel_per_batch, num_features, shared_mem);

    if (threadIdx.x == 0) {
        const float rms = sqrtf(sumsq / num_features + eps);
        shared_mem[0] = __fdiv_rd(1.0f, rms);
    }
    __syncthreads();

    const float scale = shared_mem[0];
    const int features_per_thread = (num_features + blockDim.x - 1) / blockDim.x;
    const int start = threadIdx.x * features_per_thread;
    const int end = min(start + features_per_thread, num_features);

    for (int feat = start; feat < end; ++feat) {
        const int idx = base_offset + feat * numel_per_batch;
        output[idx] = static_cast<scalar_t>(static_cast<float>(input[idx]) * scale);
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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<total_blocks, threads_per_block>>>(
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