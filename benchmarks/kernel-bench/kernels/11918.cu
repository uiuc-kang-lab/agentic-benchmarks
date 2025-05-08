#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized kernel: each block processes one batch element. Reads are performed using __ldg() for read-only global memory accesses.
// Assumes the input tensors are 128-bit aligned. Threads cooperate in shared memory to reduce the sum of squared differences.

template <typename scalar_t>
__global__ void triplet_margin_loss_kernel_optimized(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* output,
    const float margin,
    const int feat_size) {

    // Each block handles one batch element.
    const int batch_idx = blockIdx.x;
    const int offset = batch_idx * feat_size;

    // Each thread will accumulate a partial sum.
    int tid = threadIdx.x;
    scalar_t sum_pos = 0;
    scalar_t sum_neg = 0;

    // Stride loop: each thread processes multiple features.
    for (int i = tid; i < feat_size; i += blockDim.x) {
        // Use __ldg for read-only memory access. Assuming the pointers are aligned to 128-bit boundaries.
        scalar_t a = __ldg(&anchor[offset + i]);
        scalar_t p = __ldg(&positive[offset + i]);
        scalar_t n = __ldg(&negative[offset + i]);
        
        scalar_t diff_pos = a - p;
        scalar_t diff_neg = a - n;
        sum_pos += diff_pos * diff_pos;
        sum_neg += diff_neg * diff_neg;
    }

    // Allocate shared memory for reduction. First half for pos, second half for neg.
    extern __shared__ scalar_t sdata[];  // size = 2 * blockDim.x
    sdata[tid] = sum_pos;
    sdata[blockDim.x + tid] = sum_neg;
    __syncthreads();

    // Parallel reduction within each block.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        scalar_t dist_pos = sqrt(sdata[0]);
        scalar_t dist_neg = sqrt(sdata[blockDim.x]);
        scalar_t loss = max(scalar_t(0), dist_pos - dist_neg + margin);
        output[batch_idx] = loss;
    }
}

// CUDA interface function

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {
  TORCH_CHECK(anchor.device().is_cuda(), "anchor must be a CUDA tensor");
  TORCH_CHECK(positive.device().is_cuda(), "positive must be a CUDA tensor");
  TORCH_CHECK(negative.device().is_cuda(), "negative must be a CUDA tensor");
  
  const int batch_size = anchor.size(0);
  const int feat_size = anchor.size(1);
  auto output = torch::zeros({batch_size}, anchor.options());
  
  // Launch one block per batch element with 256 threads
  const int threads = 256;
  const dim3 grid(batch_size);
  // Allocate shared memory: 2 arrays of blockDim.x each, sized in bytes
  const size_t shared_mem = threads * 2 * sizeof(anchor.scalar_type().type());

  AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel_optimized", ([&] {
    triplet_margin_loss_kernel_optimized<scalar_t><<<grid, threads, threads * 2 * sizeof(scalar_t)>>>(
        anchor.data_ptr<scalar_t>(),
        positive.data_ptr<scalar_t>(),
        negative.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        margin,
        feat_size);
  }));
  
  return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &triplet_margin_loss_cuda, "Triplet margin loss forward (CUDA, optimized)");
}
