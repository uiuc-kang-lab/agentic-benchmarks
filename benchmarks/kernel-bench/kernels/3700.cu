#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that computes the HardSigmoid activation and performs an intra-block reduction
// to accumulate the sum of the activated values using shared memory and warp-level primitives.

template <typename scalar_t>
__global__ void hardsigmoid_reduce_kernel(const scalar_t* __restrict__ input,
                                            scalar_t* __restrict__ output,
                                            scalar_t* __restrict__ block_sums,
                                            size_t numel) {
    extern __shared__ scalar_t sdata[];  // Shared memory for warp-level reduction results
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    scalar_t thread_sum = 0;

    // Process elements in a grid-stride loop
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t y = (x + static_cast<scalar_t>(3)) * static_cast<scalar_t>(0.16666667);
        y = (y < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0)
             : ((y > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1) : y);
        output[i] = y;
        thread_sum += y;
    }

    // Perform warp-level reduction using __shfl_down_sync
    unsigned mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Each warp's first thread writes its reduced sum to shared memory
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    // Now, reduce the sums from each warp in the block using the first few threads
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < num_warps) {
        scalar_t val = sdata[threadIdx.x];
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (threadIdx.x == 0) {
            block_sums[blockIdx.x] = val;
        }
    }
}

// Final reduction kernel that aggregates per-block sums into a single scalar
template <typename scalar_t>
__global__ void final_reduce_kernel(const scalar_t* __restrict__ block_sums,
                                      scalar_t* __restrict__ final_sum,
                                      int num_blocks) {
    extern __shared__ scalar_t sdata[];
    int tid = threadIdx.x;
    scalar_t sum = 0;
    // Each thread sums over several elements of the block_sums array
    for (int i = tid; i < num_blocks; i += blockDim.x) {
         sum += block_sums[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Standard reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s /= 2) {
         if (tid < s) {
             sdata[tid] += sdata[tid + s];
         }
         __syncthreads();
    }
    if (tid == 0) {
         final_sum[0] = sdata[0];
    }
}

// The forward function applies the HardSigmoid activation and concurrently computes a reduction
// of the activated values. The activation output (with same shape as input) is returned.
// The reduction result is stored in an auxiliary tensor and can be used for further computations if desired.

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    // Temporary tensor to hold per-block reduction sums
    auto block_sum_tensor = torch::empty({blocks}, input.options());
    // Tensor to hold the final reduced sum (a single scalar)
    auto final_sum_tensor = torch::empty({1}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_reduce_cuda", ([&] {
        // Allocate shared memory: one element per warp in the block
        int shared_mem_bytes = ((threads + warpSize - 1) / warpSize) * sizeof(scalar_t);
        hardsigmoid_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_bytes>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_sum_tensor.data_ptr<scalar_t>(),
            numel
        );

        // If more than one block is used, launch a kernel to reduce block sums into a final sum
        if (blocks > 1) {
            int final_threads = 256;
            int final_shared_mem = final_threads * sizeof(scalar_t);
            final_reduce_kernel<scalar_t><<<1, final_threads, final_shared_mem>>>(
                block_sum_tensor.data_ptr<scalar_t>(),
                final_sum_tensor.data_ptr<scalar_t>(),
                blocks
            );
        } else {
            // For a single block, copy the block sum directly
            cudaMemcpy(final_sum_tensor.data_ptr<scalar_t>(),
                       block_sum_tensor.data_ptr<scalar_t>(),
                       sizeof(scalar_t), cudaMemcpyDeviceToDevice);
        }
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    // The activation output is correct, and the reduced sum (in final_sum_tensor[0]) is available if needed.
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation with reduction optimization (CUDA)");
}
