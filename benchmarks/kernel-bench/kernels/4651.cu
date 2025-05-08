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
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    extern __shared__ scalar_t shared_mem[];
    scalar_t* shared_sumsq = shared_mem;
    
    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        for (int offset_in_batch = 0; offset_in_batch < numel_per_batch; offset_in_batch++) {
            if (offset_in_batch % stride == tid / blockDim.x) {
                const int batch_offset = batch_id * num_features * numel_per_batch;
                
                // Calculate sum of squares for this position
                scalar_t sumsq = 0.0f;
                for (int feat = local_tid; feat < num_features; feat += blockDim.x) {
                    const scalar_t val = input[batch_offset + feat * numel_per_batch + offset_in_batch];
                    sumsq += val * val;
                }
                
                // Store in shared memory
                shared_sumsq[local_tid] = sumsq;
                __syncthreads();
                
                // Parallel reduction in shared memory
                for (int s = blockDim.x/2; s > 0; s >>= 1) {
                    if (local_tid < s) {
                        shared_sumsq[local_tid] += shared_sumsq[local_tid + s];
                    }
                    __syncthreads();
                }
                
                // Calculate RMS
                scalar_t rms;
                if (local_tid == 0) {
                    rms = sqrt(shared_sumsq[0] / num_features + eps);
                    shared_sumsq[0] = rms;  // Store for other threads to use
                }
                __syncthreads();
                
                rms = shared_sumsq[0];  // All threads read the computed RMS
                
                // Normalize the values
                for (int feat = local_tid; feat < num_features; feat += blockDim.x) {
                    const int out_idx = batch_offset + feat * numel_per_batch + offset_in_batch;
                    output[out_idx] = input[out_idx] / rms;
                }
            }
        }
    }
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

    const int threads_per_block = 256;
    const int total_elements = batch_size * numel_per_batch;
    
    int device_id;
    cudaGetDevice(&device_id);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    const int blocks = min(65535, sm_count * 32);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<blocks, threads_per_block, 0, stream>>>(
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