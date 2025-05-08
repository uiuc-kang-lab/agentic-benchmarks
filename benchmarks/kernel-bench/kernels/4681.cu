#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    __shared__ float shared_sums[32];
    
    const int element_idx = blockIdx.x;
    const int batch_id = element_idx / numel_per_batch;
    const int offset_in_batch = element_idx % numel_per_batch;
    
    if (batch_id >= batch_size) return;

    const int base_offset = batch_id * num_features * numel_per_batch + offset_in_batch;
    float sum_sq = 0.0f;

    // Feature accumulation with stride
    for (int feat = threadIdx.x; feat < num_features; feat += blockDim.x) {
        const scalar_t val = input[base_offset + feat * numel_per_batch];
        sum_sq += static_cast<float>(val) * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Store warp sums
    if (threadIdx.x % 32 == 0) {
        shared_sums[threadIdx.x / 32] = sum_sq;
    }
    __syncthreads();

    // Final cross-warp reduction
    if (threadIdx.x < 32) {
        sum_sq = threadIdx.x < (blockDim.x + 31)/32 ? shared_sums[threadIdx.x] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        
        if (threadIdx.x == 0) {
            const scalar_t rms = sqrtf(sum_sq / num_features + eps);
            shared_sums[0] = rms;
        }
    }
    __syncthreads();

    const scalar_t rms_inv = 1.0 / shared_sums[0];

    // Write normalized values
    for (int feat = threadIdx.x; feat < num_features; feat += blockDim.x) {
        const int idx = base_offset + feat * numel_per_batch;
        output[idx] = input[idx] * rms_inv;
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

    const int element_count = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<element_count, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA) optimized with warp reductions");
}