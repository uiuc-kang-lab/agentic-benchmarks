#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel function for RMS normalization
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
    const int stride = blockDim.x * gridDim.x;
    const int total_elements = batch_size * numel_per_batch;
    
    for (int idx = tid; idx < total_elements; idx += stride) {
        const int batch_id = idx / numel_per_batch;
        const int offset_in_batch = idx % numel_per_batch;
        const int batch_offset = batch_id * num_features * numel_per_batch;
        
        // Calculate sum of squares using grid-stride loop
        scalar_t sumsq = 0.0f;
        #pragma unroll
        for (int feat = 0; feat < num_features; feat++) {
            const scalar_t val = input[batch_offset + feat * numel_per_batch + offset_in_batch];
            sumsq += val * val;
        }
        
        // Calculate RMS
        const scalar_t rms = sqrt(sumsq / num_features + eps);
        
        // Normalize using grid-stride loop
        #pragma unroll
        for (int feat = 0; feat < num_features; feat++) {
            const int out_idx = batch_offset + feat * numel_per_batch + offset_in_batch;
            output[out_idx] = input[out_idx] / rms;
        }
    }
}

// Host function to launch the kernel with streams
torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    int numel_per_batch = 1;
    for(int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    const int threads_per_block = 256;
    const int max_blocks = 65535;
    const int total_elements = batch_size * numel_per_batch;
    const int blocks = min(max_blocks, (total_elements + threads_per_block - 1) / threads_per_block);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

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

    // Synchronize the stream
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA)");
}