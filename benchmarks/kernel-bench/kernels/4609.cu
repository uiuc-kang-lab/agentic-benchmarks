#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// Kernel that computes RMSNorm for a chunk of the input tensor, where the input pointer
// corresponds to a contiguous block of local_batch rows (each of shape [num_features, numel_per_batch]).
// It uses stride loops to cover all elements in the chunk.

template <typename scalar_t>
__global__ void async_rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int local_batch,      // number of batches in this chunk
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = local_batch * numel_per_batch;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < total_work; i += stride) {
        int batch_id = i / numel_per_batch;
        int offset = i % numel_per_batch;
        int base = batch_id * num_features * numel_per_batch;

        scalar_t sumsq = static_cast<scalar_t>(0);
        for (int feat = 0; feat < num_features; feat++) {
            int idx = base + feat * numel_per_batch + offset;
            scalar_t val = input[idx];
            sumsq += val * val;
        }

        scalar_t rms = sqrt(sumsq / num_features + eps);

        for (int feat = 0; feat < num_features; feat++) {
            int idx = base + feat * numel_per_batch + offset;
            output[idx] = input[idx] / rms;
        }
    }
}

// Forward function that overlaps host-device memory transfers with kernel execution
// using CUDA streams. Assumes the input tensor is on CPU and uses pinned memory
// to allow for asynchronous transfers.

torch::Tensor rms_norm_cuda_forward_async(torch::Tensor input, float eps) {
    // Ensure input is in pinned memory
    if (!input.is_pinned()) {
        input = input.pin_memory();
    }
    // Allocate output tensor in pinned memory
    auto output = torch::empty_like(input, input.options().pinned_memory(true));

    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }
    const int elements_per_batch = num_features * numel_per_batch;

    // Choose a chunk size along the batch dimension to pipeline transfers and computation
    const int CHUNK_BATCH = 16;
    const int num_chunks = (batch_size + CHUNK_BATCH - 1) / CHUNK_BATCH;

    // Use 2 CUDA streams for overlapping memory copies and kernel execution
    const int NUM_STREAMS = 2;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamCreate(&streams[s]);
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_async", ([&] {
        using scalar = scalar_t;
        // Allocate device buffers for each stream. Each buffer holds the maximum number
        // of elements for a chunk: CHUNK_BATCH * (num_features * numel_per_batch).
        scalar* d_in[NUM_STREAMS];
        scalar* d_out[NUM_STREAMS];
        size_t max_chunk_elems = CHUNK_BATCH * elements_per_batch;
        size_t max_chunk_bytes = max_chunk_elems * sizeof(scalar);
        for (int s = 0; s < NUM_STREAMS; s++) {
            cudaMalloc(&d_in[s], max_chunk_bytes);
            cudaMalloc(&d_out[s], max_chunk_bytes);
        }

        // Process each chunk: asynchronously copy from host to device, launch kernel, and copy back
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int start_batch = chunk * CHUNK_BATCH;
            int local_batch = std::min(CHUNK_BATCH, batch_size - start_batch);
            int chunk_elems = local_batch * elements_per_batch;
            size_t chunk_bytes = chunk_elems * sizeof(scalar);
            int stream_id = chunk % NUM_STREAMS;
            cudaStream_t stream = streams[stream_id];

            // Compute pointers for the current chunk in the pinned host memory
            const scalar* h_in = input.data_ptr<scalar>() + start_batch * elements_per_batch;
            scalar* h_out = output.data_ptr<scalar>() + start_batch * elements_per_batch;

            // Asynchronously copy input chunk from host to device
            cudaMemcpyAsync(d_in[stream_id], h_in, chunk_bytes, cudaMemcpyHostToDevice, stream);

            // Launch the RMSNorm kernel for this chunk
            int total_work = local_batch * numel_per_batch;
            int threads_per_block = 256;
            int blocks = (total_work + threads_per_block - 1) / threads_per_block;
            async_rms_norm_kernel<scalar><<<blocks, threads_per_block, 0, stream>>>(
                d_in[stream_id],
                d_out[stream_id],
                local_batch,
                num_features,
                numel_per_batch,
                eps
            );

            // Asynchronously copy the computed output chunk from device to host
            cudaMemcpyAsync(h_out, d_out[stream_id], chunk_bytes, cudaMemcpyDeviceToHost, stream);
        }

        // Synchronize all streams to ensure processing is complete
        for (int s = 0; s < NUM_STREAMS; s++) {
            cudaStreamSynchronize(streams[s]);
            cudaFree(d_in[s]);
            cudaFree(d_out[s]);
            cudaStreamDestroy(streams[s]);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward_async, "RMS normalization forward with async pipelining (CUDA)");
}
