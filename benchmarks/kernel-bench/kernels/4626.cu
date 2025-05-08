#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constant memory for storing epsilon
__constant__ float d_eps;

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch
) {
    extern __shared__ char shared_mem[];
    scalar_t* shared_sumsq = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = tid / numel_per_batch;
    
    if (batch_id >= batch_size) return;
    
    const int offset_in_batch = tid % numel_per_batch;
    const int batch_offset = batch_id * num_features * numel_per_batch;
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    // Calculate sum of squares using shared memory for partial sums
    scalar_t thread_sumsq = 0.0f;
    for (int feat = 0; feat < num_features; feat++) {
        const scalar_t val = input[batch_offset + feat * numel_per_batch + offset_in_batch];
        thread_sumsq += val * val;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sumsq += __shfl_down_sync(0xffffffff, thread_sumsq, offset);
    }

    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_sumsq[warp_id] = thread_sumsq;
    }
    __syncthreads();

    // Final reduction across warps (done by first warp)
    if (warp_id == 0 && lane_id < (blockDim.x / 32)) {
        scalar_t warp_sumsq = shared_sumsq[lane_id];
        #pragma unroll
        for (int offset = (blockDim.x / 64); offset > 0; offset /= 2) {
            warp_sumsq += __shfl_down_sync(0xffffffff, warp_sumsq, offset);
        }
        shared_sumsq[0] = warp_sumsq;
    }
    __syncthreads();

    // Calculate RMS using the final sum
    const scalar_t rms = sqrt(shared_sumsq[0] / num_features + d_eps);
    
    // Normalize
    for (int feat = 0; feat < num_features; feat++) {
        const int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
        output[idx] = input[idx] / rms;
    }
}

void set_constant_eps(float eps) {
    cudaMemcpyToSymbol(d_eps, &eps, sizeof(float));
}

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    // Calculate elements per batch for dimensions beyond first two
    int numel_per_batch = 1;
    for(int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    const int total_threads = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // Set constant memory
    set_constant_eps(eps);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            numel_per_batch
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA)");
}