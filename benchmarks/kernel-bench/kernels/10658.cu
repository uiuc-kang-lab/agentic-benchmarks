#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel uses a warp-level inclusive scan to compute the cumulative product
// for each sequence. Each block processes one (or more via grid-stride) sequence,
// and within a sequence the work is divided among warps. The per-warp results are
// combined in shared memory to produce the final cumulative product, reducing the
// dependency chain from O(n) to O(log n).

template <typename scalar_t>
__global__ void cumprod_kernel_warp_scan(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t total_sequences,
    const int64_t dim_size,
    const int64_t stride) {

  // Block's thread index
  int tid = threadIdx.x;
  // Lane and warp identification
  int lane = tid & 31;
  int warpId = tid >> 5;
  
  // Process sequences in a grid-stride loop
  for (int seq = blockIdx.x; seq < total_sequences; seq += gridDim.x) {
    // Compute the base offset for this sequence using the original indexing scheme
    int batch_idx = seq / stride;
    int in_idx = seq % stride;
    int64_t base_offset = batch_idx * (stride * dim_size) + in_idx;
    
    // Each thread loads one element of the sequence if in bounds; else use identity (1)
    scalar_t x = (tid < dim_size) ? input[base_offset + tid * stride] : (scalar_t)1;
    
    // Perform warp-level inclusive scan using shuffle intrinsics
    // Each warp computes the cumulative product for its segment
    for (int offset = 1; offset < 32; offset *= 2) {
      scalar_t n = __shfl_up_sync(0xffffffff, x, offset, 32);
      if (lane >= offset) {
        x *= n;
      }
    }
    
    // Allocate shared memory to store the per-warp cumulative product (the last lane of each warp)
    __shared__ scalar_t warpProducts[32];  // supports up to 1024 threads per block / 32 = 32 warps
    if (lane == 31) {
      warpProducts[warpId] = x;
    }
    __syncthreads();
    
    // Let the first warp combine the per-warp results
    // Since the number of warps is small, a single thread can combine them
    if (warpId == 0) {
      int numWarps = (blockDim.x + 31) / 32;
      // Only one thread (e.g., lane 0) performs this sequential update
      if (tid == 0) {
        for (int i = 1; i < numWarps; i++) {
          warpProducts[i] *= warpProducts[i - 1];
        }
      }
    }
    __syncthreads();
    
    // Each warp (except the first) multiplies its scanned values by the product of all previous warps
    if (warpId > 0 && tid < dim_size) {
      x *= warpProducts[warpId - 1];
    }
    
    // Write the computed cumulative product back to global memory
    if (tid < dim_size) {
      output[base_offset + tid * stride] = x;
    }
    __syncthreads(); // Ensure shared memory is ready before processing next sequence
  }
}


torch::Tensor cumprod_cuda_forward_warp(torch::Tensor input, int64_t dim) {
  auto output = torch::empty_like(input);
  
  // Retrieve tensor properties
  auto sizes = input.sizes();
  auto strides = input.strides();
  
  int64_t dim_size = sizes[dim];
  int64_t stride = strides[dim];
  int64_t numel = input.numel();
  int64_t total_sequences = numel / dim_size;
  
  // Determine the number of threads per block as the next multiple of 32 covering dim_size
  int threads_per_block = ((dim_size + 31) / 32) * 32;
  if (threads_per_block > 1024) {
    threads_per_block = 1024; // fallback if dim_size is very large
  }
  
  // Choose grid size. Here we use up to 256 blocks; grid-stride loop handles extra sequences.
  int blocks = total_sequences < 256 ? total_sequences : 256;
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_warp", ([&] {
    cumprod_kernel_warp_scan<scalar_t><<<blocks, threads_per_block>>>(
      output.data_ptr<scalar_t>(),
      input.data_ptr<scalar_t>(),
      total_sequences,
      dim_size,
      stride
    );
  }));
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cumprod_cuda_forward_warp, "Cumulative product forward using warp-scan (CUDA)");
}
