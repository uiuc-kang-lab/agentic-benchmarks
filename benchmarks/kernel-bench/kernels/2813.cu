#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstring>

// Use 256 threads per block and process CHUNK_SIZE elements per chunk
const int THREADS = 256;
// Process 1<<20 elements per chunk (~1M elements)
const int CHUNK_SIZE = 1 << 20;  

// Kernel to compute sigmoid on a chunk of data
// Each thread processes one element
template <typename scalar_t>
__global__ void sigmoid_chunk_kernel(const scalar_t* __restrict__ in,
                                       scalar_t* __restrict__ out,
                                       int chunk_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < chunk_size) {
    // Compute sigmoid: 1/(1+exp(-x)) with proper cast
    float val = static_cast<float>(in[idx]);
    out[idx] = static_cast<scalar_t>(1.0f / (1.0f + expf(-val)));
  }
}


// Forward function that overlaps memory transfers with kernel execution
// Assumes input is a CPU tensor. The pipeline transfers data to GPU,
// runs the kernel, and copies the result back, overlapping these steps

torch::Tensor forward(torch::Tensor input) {
  // Check that input is on CPU
  TORCH_CHECK(!input.is_cuda(), "Expected input tensor to be on CPU");
  
  // Get total number of elements
  const int64_t N = input.numel();

  // Allocate output tensor on CPU
  auto output = torch::empty_like(input);

  // Dispatch based on the scalar type (float/double etc.)
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "stream_overlap_sigmoid_forward", ([&] {
    using scalar_t = scalar_t;

    // Allocate pinned host memory for the entire input
    scalar_t* pinned_input = nullptr;
    cudaHostAlloc((void**)&pinned_input, N * sizeof(scalar_t), cudaHostAllocDefault);
    // Copy input data from normal host memory to pinned memory
    std::memcpy(pinned_input, input.data_ptr<scalar_t>(), N * sizeof(scalar_t));

    // Setup 2 CUDA streams for overlapping computation and transfers
    int n_streams = 2;
    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) {
      cudaStreamCreate(&streams[i]);
    }

    // Allocate device buffers and pinned output buffers for each stream
    std::vector<scalar_t*> d_in(n_streams, nullptr);
    std::vector<scalar_t*> d_out(n_streams, nullptr);
    std::vector<scalar_t*> pinned_out(n_streams, nullptr);
    for (int i = 0; i < n_streams; i++) {
      cudaMalloc((void**)&d_in[i], CHUNK_SIZE * sizeof(scalar_t));
      cudaMalloc((void**)&d_out[i], CHUNK_SIZE * sizeof(scalar_t));
      cudaHostAlloc((void**)&pinned_out[i], CHUNK_SIZE * sizeof(scalar_t), cudaHostAllocDefault);
    }

    // Compute number of chunks
    int64_t num_chunks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;
    // For each stream, record the offset and chunk size of the last submitted work
    std::vector<int64_t> stream_chunk_offset(n_streams, -1);
    std::vector<int> stream_chunk_size(n_streams, 0);

    // Process each chunk in a pipelined fashion
    for (int64_t i = 0; i < num_chunks; i++) {
      int stream_id = i % n_streams;
      int64_t offset = i * CHUNK_SIZE;
      int current_chunk_size = static_cast<int>(std::min((int64_t)CHUNK_SIZE, N - offset));
      
      // If this stream was used before, wait for it to complete and copy its output
      if (stream_chunk_offset[stream_id] != -1) {
        cudaStreamSynchronize(streams[stream_id]);
        std::memcpy(output.data_ptr<scalar_t>() + stream_chunk_offset[stream_id],
                    pinned_out[stream_id],
                    stream_chunk_size[stream_id] * sizeof(scalar_t));
      }
      
      // Asynchronously copy the current chunk from pinned host memory to device
      cudaMemcpyAsync(d_in[stream_id], pinned_input + offset,
                      current_chunk_size * sizeof(scalar_t),
                      cudaMemcpyHostToDevice, streams[stream_id]);
      
      // Launch the sigmoid kernel on this chunk
      int blocks = (current_chunk_size + THREADS - 1) / THREADS;
      sigmoid_chunk_kernel<scalar_t><<<blocks, THREADS, 0, streams[stream_id]>>>(d_in[stream_id], d_out[stream_id], current_chunk_size);
      
      // Asynchronously copy the result from device to the pinned output buffer
      cudaMemcpyAsync(pinned_out[stream_id], d_out[stream_id],
                      current_chunk_size * sizeof(scalar_t),
                      cudaMemcpyDeviceToHost, streams[stream_id]);
      
      // Record this chunk's offset and size for later copying into the final output
      stream_chunk_offset[stream_id] = offset;
      stream_chunk_size[stream_id] = current_chunk_size;
    }

    // For each stream, copy any remaining output data
    for (int stream_id = 0; stream_id < n_streams; stream_id++) {
      if (stream_chunk_offset[stream_id] != -1) {
        cudaStreamSynchronize(streams[stream_id]);
        std::memcpy(output.data_ptr<scalar_t>() + stream_chunk_offset[stream_id],
                    pinned_out[stream_id],
                    stream_chunk_size[stream_id] * sizeof(scalar_t));
      }
    }

    // Cleanup: free device buffers, pinned buffers, and destroy streams
    for (int i = 0; i < n_streams; i++) {
      cudaFree(d_in[i]);
      cudaFree(d_out[i]);
      cudaFreeHost(pinned_out[i]);
      cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(pinned_input);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Sigmoid forward with overlapped memory transfers using CUDA streams");
}
