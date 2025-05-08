#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rms_norm_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    extern __shared__ float shmem[];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int spatial_idx = blockIdx.y;
    const int batch_id = blockIdx.z;

    if (batch_id >= batch_size) return;
    
    const int base_offset = batch_id * num_features * numel_per_batch + spatial_idx;
    
    // Load data into shared memory
    float local_sum = 0.0f;
    for (int feat = threadIdx.x; feat < num_features; feat += blockDim.x) {
        const scalar_t val = input[base_offset + feat * numel_per_batch];
        local_sum += static_cast<float>(val) * static_cast<float>(val);
    }
    shmem[threadIdx.x] = local_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float rms = sqrtf(shmem[0] / num_features + eps);
    const scalar_t inv_rms = 1.0f / rms;

    // Normalize with vectorized stores
    for (int feat = threadIdx.x; feat < num_features; feat += blockDim.x) {
        const int idx = base_offset + feat * numel_per_batch;
        output[idx] = input[idx] * inv_rms;
    }
}

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    int numel_per_batch = 1;
    for(int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    dim3 blocks(num_features, numel_per_batch, batch_size);
    const int threads = 256;
    size_t shmem_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_shared_kernel<scalar_t><<<blocks, threads, shmem_size>>>(
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
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA) with shared memory");
}
