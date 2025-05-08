#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

// Kernel to compute argmax using stride loops and warp-level intrinsics for reduction
__global__ void argmax_stride_loop_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    const int total = outerSize * innerSize;
    const int warpSize = 32;

    // Loop over outer*inner pairs using a grid-stride loop
    for (int idx = blockIdx.x; idx < total; idx += gridDim.x) {
        // Determine corresponding outer and inner indices
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;
        int start_offset = outer_idx * dimSize * innerSize + inner_idx;

        // Each thread processes part of the 'dim' dimension with a stride loop
        float local_max = -FLT_MAX;
        int local_idx = 0;
        for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
            float val = __ldg(&x[start_offset + d * innerSize]);
            #pragma unroll 4  // Unroll the loop for better instruction-level parallelism
            if (val > local_max) {
                local_max = val;
                local_idx = d;
            }
        }

        // Warp-level reduction using shuffle intrinsics to combine thread results
        unsigned int mask = 0xffffffff;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(mask, local_max, offset);
            int other_idx = __shfl_down_sync(mask, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        // Allocate shared memory for inter-warp reduction
        extern __shared__ char shared_mem[]; 
        float* warp_max = (float*)shared_mem;
        int* warp_arg = (int*)(shared_mem + ((blockDim.x + warpSize - 1) / warpSize) * sizeof(float));

        int lane = threadIdx.x & (warpSize - 1);
        int warp_id = threadIdx.x / warpSize;
        if (lane == 0) {
            warp_max[warp_id] = local_max;
            warp_arg[warp_id] = local_idx;
        }
        __syncthreads();

        // Final reduction across warps, performed by a single thread
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        if (threadIdx.x == 0) {
            float final_max = warp_max[0];
            int final_idx = warp_arg[0];
            for (int i = 1; i < numWarps; i++) {
                float candidate = warp_max[i];
                int candidate_idx = warp_arg[i];
                if (candidate > final_max) {
                    final_max = candidate;
                    final_idx = candidate_idx;
                }
            }
            indices[idx] = final_idx;
        }
        __syncthreads(); // Ensure all threads are synchronized before next iteration
    }
}

// Host function to launch the CUDA kernel
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dimension for argmax.");

    int outerSize = 1;
    for (int i = 0; i < dim; i++) {
        outerSize *= sizes[i];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int i = dim + 1; i < ndim; i++) {
        innerSize *= sizes[i];
    }

    // Build the output shape by removing the 'dim' dimension
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            out_sizes.push_back(sizes[i]);
        }
    }

    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Launch configuration
    const int threads = 256;
    int total = outerSize * innerSize;
    int blocks = (total < 1024) ? total : 1024;  // Cap the number of blocks to 1024
    
    // Calculate shared memory required for warp reduction
    int nWarps = (threads + 31) / 32;
    size_t shared_mem_size = nWarps * (sizeof(float) + sizeof(int));

    argmax_stride_loop_kernel<<<blocks, threads, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize);

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (stride-loop with warp shuffle reduction)");
}
