#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

// Forward function that partitions the batch and overlaps computation via CUDA streams
// using asynchronous kernel launches on different streams
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  if (bias.has_value()) {
    CHECK_INPUT(*bias);
  }

  // Partition the batch dimension to overlap computation with memory operations
  int64_t batch_size = x.size(0);
  int num_streams = 2; // Using 2 streams, can be adjusted based on workload and GPU capability
  int64_t chunk_size = (batch_size + num_streams - 1) / num_streams;

  std::vector<torch::Tensor> output_chunks;
  std::vector<cudaStream_t> streams(num_streams);

  // Create CUDA streams
  for (int i = 0; i < num_streams; i++) {
    cudaError_t err = cudaStreamCreate(&streams[i]);
    TORCH_CHECK(err == cudaSuccess, "Failed to create CUDA stream");
  }

  // Launch operations concurrently for each batch chunk
  for (int i = 0; i < num_streams; i++) {
    int64_t start_idx = i * chunk_size;
    if (start_idx >= batch_size)
      break;
    int64_t end_idx = std::min(start_idx + chunk_size, batch_size);

    // Slice the input tensor for the current batch chunk
    torch::Tensor x_chunk = x.slice(0, start_idx, end_idx);

    // Set the CUDA stream for the following operations
    {
      at::cuda::CUDAStreamGuard guard = at::cuda::getCurrentCUDAStream();
      // Launch the transposed convolution on this chunk asynchronously
      torch::Tensor out_chunk = at::conv_transpose3d(
          x_chunk,
          weight,
          bias.has_value() ? *bias : at::Tensor(),
          stride,
          padding,
          output_padding,
          groups);
      output_chunks.push_back(out_chunk);
    }
  }

  // Synchronize all streams to ensure computation and any implicit memory transfers are complete
  for (int i = 0; i < num_streams; i++) {
    cudaError_t err = cudaStreamSynchronize(streams[i]);
    TORCH_CHECK(err == cudaSuccess, "Failed to synchronize CUDA stream");
    cudaStreamDestroy(streams[i]);
  }

  // Concatenate the output chunks along the batch dimension
  torch::Tensor output = torch::cat(output_chunks, 0);
  return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward with overlapped CUDA streams");
}
