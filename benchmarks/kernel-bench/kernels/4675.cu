#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int NUM_STREAMS = 4;

template <typename scalar_t>
__global__ void rms_norm_kernel_streamed(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps,
    const int stream_idx
) {
    const int batches_per_stream = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS; const int total_threads = batches_per_stream * numel_per_batch;
    const int start_batch = stream_idx * batches_per_stream;
    const int end_batch = min(start_batch + batches_per_stream, batch_size);
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = start_batch + (tid / numel_per_batch);
    
    if (batch_id >= end_batch) return;
    
    const int offset_in_batch = tid % numel_per_batch;
    const int batch_offset = (batch_id - start_batch) * num_features * numel_per_batch;

    // Calculate sum of squares
    scalar_t sumsq = 0.0f;
    for (int feat = 0; feat < num_features; feat++) {
        const scalar_t val = input[batch_id * num_features * numel_per_batch + feat * numel_per_batch + offset_in_batch];
        sumsq += val * val;
    }
    
    // Calculate RMS
    const scalar_t rms = sqrt(sumsq / num_features + eps);
    
    // Normalize
    for (int feat = 0; feat < num_features; feat++) {
        const int idx = batch_id * num_features * numel_per_batch + feat * numel_per_batch + offset_in_batch;
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

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads_per_block = 256;
    const int batches_per_stream = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS; const int total_threads = batches_per_stream * numel_per_batch;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            const int total_threads = batches_per_stream * numel_per_batch;
            const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
            
            rms_norm_kernel_streamed<scalar_t><<<blocks, threads_per_block, 0, streams[i]>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                num_features,
                numel_per_batch,
                eps,
                i
            );
        }
    }));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA) with stream overlap");
}