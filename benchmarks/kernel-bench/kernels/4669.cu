#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rms_norm_optimized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    __shared__ float shared_sum[256];

    const int batch_element_id = blockIdx.x;
    const int batch_id = batch_element_id / numel_per_batch;
    const int offset_in_batch = batch_element_id % numel_per_batch;
    
    if (batch_id >= batch_size) return;

    const int feature_idx = threadIdx.x;
    const int features_per_thread = (num_features + blockDim.x - 1) / blockDim.x;
    const int start_feature = feature_idx * features_per_thread;
    const int end_feature = min(start_feature + features_per_thread, num_features);

    float local_sum = 0.0f;
    const int base_offset = batch_id * num_features * numel_per_batch + offset_in_batch;
    
    for (int feat = start_feature; feat < end_feature; ++feat) {
        const scalar_t val = input[base_offset + feat * numel_per_batch];
        local_sum = __fmaf_rn(static_cast<float>(val), static_cast<float>(val), local_sum);
    }

    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // Warp reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float rms = sqrtf(shared_sum[0] / num_features + eps);
        shared_sum[0] = rms;
    }
    __syncthreads();

    const scalar_t rms = static_cast<scalar_t>(shared_sum[0]);

    for (int feat = start_feature; feat < end_feature; ++feat) {
        const int idx = base_offset + feat * numel_per_batch;
        output[idx] = static_cast<scalar_t>(input[idx] / rms);
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
        rms_norm_optimized_kernel<scalar_t><<<total_blocks, threads_per_block>>>(
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