#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>

// Kernel that processes a chunk of the outer dimension
// outer_offset is added to blockIdx.y to get the global outer index
// Each thread computes the argmin along the K dimension for a given (outer, inner) pair

template <typename scalar_t>
__global__ void argmin_pipeline_kernel(const scalar_t* __restrict__ x,
                                         int64_t* __restrict__ output,
                                         int K,
                                         int64_t inner_size,
                                         int64_t outer_offset) {
  int outer = blockIdx.y + outer_offset;
  int inner = blockIdx.x * blockDim.x + threadIdx.x;
  if (inner >= inner_size) return;

  // Compute the start of the slice corresponding to the (outer, inner) pair
  const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;
  scalar_t min_val = slice_start[0];
  int min_index = 0;
  for (int k = 1; k < K; ++k) {
    scalar_t val = slice_start[k * inner_size];
    if (val < min_val) {
      min_val = val;
      min_index = k;
    }
  }
  output[outer * inner_size + inner] = min_index;
}

// Host function that partitions the work across multiple CUDA streams to overlap
// kernel computation with asynchronous memory transfers (device-to-host).
// The final result is copied into a pinned host tensor which is returned.

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  // Ensure input is a CUDA tensor
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

  int dims = x.dim();
  if (dim < 0) {
    dim += dims;
  }
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  // Compute outer_size, reduction dimension (K), and inner_size
  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  int K = static_cast<int>(x.size(dim));
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }

  // Determine the output shape (same as input shape with the reduction dim removed)
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }

  // Allocate output buffer on device for kernel computation
  auto dev_options = x.options().dtype(at::kLong);
  at::Tensor output_device = at::empty(out_sizes, dev_options);

  // Allocate pinned host memory for the final result to overlap the device-to-host transfer
  at::Tensor output_host = at::empty(out_sizes, at::TensorOptions()
                                              .device(at::kCPU)
                                              .dtype(at::kLong)
                                              .pinned_memory(true));

  // Get raw pointers
  int64_t* output_device_ptr = output_device.data_ptr<int64_t>();
  int64_t* output_host_ptr = output_host.data_ptr<int64_t>();

  // Launch kernels on multiple streams to overlap kernel execution and memory copies
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    const scalar_t* x_data = x.data_ptr<scalar_t>();

    const int num_streams = 4;
    const int threads = 256;
    dim3 block_dim(threads);

    // Partition the outer dimension across streams
    int64_t chunk_size = (outer_size + num_streams - 1) / num_streams;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
      cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < num_streams; i++) {
      int64_t offset = i * chunk_size;
      if (offset >= outer_size) break;
      int64_t current_chunk = std::min(chunk_size, outer_size - offset);
      dim3 grid_dim((inner_size + threads - 1) / threads, current_chunk);
      
      // Launch the kernel on the i-th stream for this chunk
      argmin_pipeline_kernel<scalar_t><<<grid_dim, block_dim, 0, streams[i]>>>(
          x_data, output_device_ptr, K, inner_size, offset);

      // Asynchronously copy the computed chunk from device to host pinned memory
      size_t bytes = current_chunk * inner_size * sizeof(int64_t);
      const int64_t* src_ptr = output_device_ptr + offset * inner_size;
      int64_t* dst_ptr = output_host_ptr + offset * inner_size;
      cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
      cudaStreamSynchronize(streams[i]);
      cudaStreamDestroy(streams[i]);
    }
  }));

  // Return the result that has been asynchronously transferred to host memory
  return output_host;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Argmin forward pipeline with overlapping computation and memory transfers (CUDA)");
}
