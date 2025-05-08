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
    const int batch_id = tid / numel_per_batch;
    
    if (batch_id >= batch_size) return;
    
    const int offset_in_batch = tid % numel_per_batch;
    const int batch_offset = batch_id * num_features * numel_per_batch;

    // Calculate sum of squares with manual unrolling for common feature sizes
    scalar_t sumsq = 0.0f;
    int feat = 0;
    
    // Unroll by 32 for bulk of computation
    #pragma unroll 32
    for (; feat < (num_features / 32) * 32; feat += 32) {
        scalar_t val0 = input[batch_offset + (feat + 0) * numel_per_batch + offset_in_batch];
        scalar_t val1 = input[batch_offset + (feat + 1) * numel_per_batch + offset_in_batch];
        scalar_t val2 = input[batch_offset + (feat + 2) * numel_per_batch + offset_in_batch];
        scalar_t val3 = input[batch_offset + (feat + 3) * numel_per_batch + offset_in_batch];
        scalar_t val4 = input[batch_offset + (feat + 4) * numel_per_batch + offset_in_batch];
        scalar_t val5 = input[batch_offset + (feat + 5) * numel_per_batch + offset_in_batch];
        scalar_t val6 = input[batch_offset + (feat + 6) * numel_per_batch + offset_in_batch];
        scalar_t val7 = input[batch_offset + (feat + 7) * numel_per_batch + offset_in_batch];
        
        sumsq += val0 * val0 + val1 * val1 + val2 * val2 + val3 * val3 +
                 val4 * val4 + val5 * val5 + val6 * val6 + val7 * val7;
        
        scalar_t val8  = input[batch_offset + (feat + 8)  * numel_per_batch + offset_in_batch];
        scalar_t val9  = input[batch_offset + (feat + 9)  * numel_per_batch + offset_in_batch];
        scalar_t val10 = input[batch_offset + (feat + 10) * numel_per_batch + offset_in_batch];
        scalar_t val11 = input[batch_offset + (feat + 11) * numel_per_batch + offset_in_batch];
        scalar_t val12 = input[batch_offset + (feat + 12) * numel_per_batch + offset_in_batch];
        scalar_t val13 = input[batch_offset + (feat + 13) * numel_per_batch + offset_in_batch];
        scalar_t val14 = input[batch_offset + (feat + 14) * numel_per_batch + offset_in_batch];
        scalar_t val15 = input[batch_offset + (feat + 15) * numel_per_batch + offset_in_batch];
        
        sumsq += val8  * val8  + val9  * val9  + val10 * val10 + val11 * val11 +
                 val12 * val12 + val13 * val13 + val14 * val14 + val15 * val15;

        scalar_t val16 = input[batch_offset + (feat + 16) * numel_per_batch + offset_in_batch];
        scalar_t val17 = input[batch_offset + (feat + 17) * numel_per_batch + offset_in_batch];
        scalar_t val18 = input[batch_offset + (feat + 18) * numel_per_batch + offset_in_batch];
        scalar_t val19 = input[batch_offset + (feat + 19) * numel_per_batch + offset_in_batch];
        scalar_t val20 = input[batch_offset + (feat + 20) * numel_per_batch + offset_in_batch];
        scalar_t val21 = input[batch_offset + (feat + 21) * numel_per_batch + offset_in_batch];
        scalar_t val22 = input[batch_offset + (feat + 22) * numel_per_batch + offset_in_batch];
        scalar_t val23 = input[batch_offset + (feat + 23) * numel_per_batch + offset_in_batch];
        
        sumsq += val16 * val16 + val17 * val17 + val18 * val18 + val19 * val19 +
                 val20 * val20 + val21 * val21 + val22 * val22 + val23 * val23;

        scalar_t val24 = input[batch_offset + (feat + 24) * numel_per_batch + offset_in_batch];
        scalar_t val25 = input[batch_offset + (feat + 25) * numel_per_batch + offset_in_batch];
        scalar_t val26 = input[batch_offset + (feat + 26) * numel_per_batch + offset_in_batch];
        scalar_t val27 = input[batch_offset + (feat + 27) * numel_per_batch + offset_in_batch];
        scalar_t val28 = input[batch_offset + (feat + 28) * numel_per_batch + offset_in_batch];
        scalar_t val29 = input[batch_offset + (feat + 29) * numel_per_batch + offset_in_batch];
        scalar_t val30 = input[batch_offset + (feat + 30) * numel_per_batch + offset_in_batch];
        scalar_t val31 = input[batch_offset + (feat + 31) * numel_per_batch + offset_in_batch];
        
        sumsq += val24 * val24 + val25 * val25 + val26 * val26 + val27 * val27 +
                 val28 * val28 + val29 * val29 + val30 * val30 + val31 * val31;
    }

    // Handle remaining elements
    #pragma unroll
    for (; feat < num_features; feat++) {
        const scalar_t val = input[batch_offset + feat * numel_per_batch + offset_in_batch];
        sumsq += val * val;
    }
    
    // Calculate RMS
    const scalar_t rms = sqrt(sumsq / num_features + eps);
    
    // Normalize with unrolled loop
    feat = 0;
    #pragma unroll 32
    for (; feat < (num_features / 32) * 32; feat += 32) {
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            const int idx = batch_offset + (feat + i) * numel_per_batch + offset_in_batch;
            output[idx] = input[idx] / rms;
        }
    }
    
    // Handle remaining elements
    #pragma unroll
    for (; feat < num_features; feat++) {
        const int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
        output[idx] = input[idx] / rms;
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

    const int total_threads = batch_size * numel_per_batch;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<blocks, threads_per_block>>>(
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