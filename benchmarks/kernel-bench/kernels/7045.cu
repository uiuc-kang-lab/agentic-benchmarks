#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <c10/cuda/CUDAStream.h>

// Kernel: Each block computes the min reduction for one output element over the reduction dimension.
// This version accepts an offset parameter so that we can process a chunk of the overall output.

template <typename scalar_t>
__global__ void min_reduce_shared_chunk_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner,
    int offset,
    int total_global) {

  int idx = blockIdx.x + offset;  // global output index
  if (idx >= total_global) return;

  int outer_idx = idx / inner;
  int inner_idx = idx % inner;
  int base = outer_idx * (r * inner) + inner_idx;

  // Allocate shared memory dynamically
  extern __shared__ char smem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
  
  int tid = threadIdx.x;
  scalar_t my_min = std::numeric_limits<scalar_t>::max();

  // Each thread reduces part of the reduction dimension
  for (int j = tid; j < r; j += blockDim.x) {
    int pos = base + j * inner;
    scalar_t val = input[pos];
    my_min = (val < my_min) ? val : my_min;
  }
  sdata[tid] = my_min;
  __syncthreads();

  // Tree reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      scalar_t other = sdata[tid + s];
      sdata[tid] = (other < sdata[tid]) ? other : sdata[tid];
    }
    __syncthreads();
  }

  // Warp-level reduction (no __syncthreads needed within a warp)
  if (tid < 32) {
    volatile scalar_t* vsdata = sdata;
    vsdata[tid] = (vsdata[tid + 32] < vsdata[tid]) ? vsdata[tid + 32] : vsdata[tid];
    vsdata[tid] = (vsdata[tid + 16] < vsdata[tid]) ? vsdata[tid + 16] : vsdata[tid];
    vsdata[tid] = (vsdata[tid + 8] < vsdata[tid]) ? vsdata[tid + 8] : vsdata[tid];
    vsdata[tid] = (vsdata[tid + 4] < vsdata[tid]) ? vsdata[tid + 4] : vsdata[tid];
    vsdata[tid] = (vsdata[tid + 2] < vsdata[tid]) ? vsdata[tid + 2] : vsdata[tid];
    vsdata[tid] = (vsdata[tid + 1] < vsdata[tid]) ? vsdata[tid + 1] : vsdata[tid];
  }
  
  if (tid == 0) {
    output[idx] = sdata[0];
  }
}

// Forward function: Splits the output tensor into chunks and uses separate CUDA streams to overlap
// kernel execution with asynchronous device-to-host memory transfers.

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate sizes: outer dimensions, reduction dimension (r), and inner dimensions
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Compute total number of output elements (after reducing 'dim')
  int total = outer * inner;

  // Determine output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }

  // Allocate intermediate output tensor on GPU (flattened)
  auto output_gpu = torch::empty({total}, input.options());

  // Allocate pinned host memory for final result to enable async DMA transfers
  auto options = torch::TensorOptions().dtype(input.dtype()).device(torch::kCPU).pinned_memory(true);
  auto output_cpu = torch::empty({total}, options);

  // Parameters for splitting the work into chunks
  int chunk_size = 1024;
  if (chunk_size > total) chunk_size = total;
  int num_chunks = (total + chunk_size - 1) / chunk_size;

  // Create CUDA streams for overlapping execution and memory copy
  std::vector<cudaStream_t> streams(num_chunks);
  for (int i = 0; i < num_chunks; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  // Launch kernels and async copy for each chunk
  int threads = (r < 256 ? r : 256);

  for (int i = 0; i < num_chunks; i++) {
    int offset = i * chunk_size;
    int current_chunk = std::min(chunk_size, total - offset);
    int blocks = current_chunk; // One block per output element in the chunk

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_shared_chunk_cuda", ([&] {
      min_reduce_shared_chunk_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t), streams[i]>>>(
          input.data_ptr<scalar_t>(),
          output_gpu.data_ptr<scalar_t>(),
          outer,
          r,
          inner,
          offset,
          total);
    }));

    // Asynchronously copy the computed chunk from device to host
    cudaMemcpyAsync(
        output_cpu.data_ptr<scalar_t>() + offset,
        output_gpu.data_ptr<scalar_t>() + offset,
        current_chunk * sizeof(scalar_t),
        cudaMemcpyDeviceToHost,
        streams[i]
    );
  }

  // Synchronize and destroy streams
  for (int i = 0; i < num_chunks; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  // Reshape the result to the expected output tensor shape
  auto output_final = output_cpu.view(torch::IntArrayRef(output_shape));
  return output_final;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction over a specified dimension with overlapped computation and memory transfers (CUDA)");
}
