#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

#define WARP_SIZE 32

// Kernel that uses grid-stride loops and internal stride loops to handle large workloads for the reduction dimension.
// Each thread processes multiple elements from the reduction dimension using a stride loop.
// Warp-level shuffle reductions are used first within each warp, then an inter-warp reduction via shared memory is applied.
// This ensures correct boundary handling when dimSize is larger than the number of threads.
__global__ void argmax_stride_loop_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    // Total number of (outer, inner) pairs
    int total = outerSize * innerSize;

    // Process the (outer, inner) pairs using a grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;
        // Compute the starting offset for the current (outer, inner) pair
        int start_offset = outer_idx * dimSize * innerSize + inner_idx;

        // Each thread uses a stride loop over the reduction dimension
        float local_max = -FLT_MAX;
        int local_arg = -1;
        for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
            float val = __ldg(&x[start_offset + d * innerSize]);
            if (val > local_max) {
                local_max = val;
                local_arg = d;
            }
        }

        // Warp-level reduction: each warp reduces its own candidates using shuffle intrinsics
        unsigned int mask = __activemask();
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(mask, local_max, offset);
            int other_arg = __shfl_down_sync(mask, local_arg, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_arg = other_arg;
            } else if (other_val == local_max && other_arg < local_arg) {
                local_arg = other_arg;
            }
        }

        // Each warp's lane 0 now contains a candidate result.
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane = threadIdx.x % WARP_SIZE;
        int nWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

        // Use dynamically allocated shared memory for inter-warp reduction
        extern __shared__ char shared_mem[];
        float* warp_max = (float*)shared_mem;                  // Space for nWarps float values
        int*   warp_arg = (int*)(shared_mem + nWarps * sizeof(float)); // Followed by nWarps int values

        if (lane == 0) {
            warp_max[warp_id] = local_max;
            warp_arg[warp_id] = local_arg;
        }
        __syncthreads();

        // Let the first warp reduce the per-warp candidates
        if (threadIdx.x < nWarps) {
            local_max = warp_max[threadIdx.x];
            local_arg = warp_arg[threadIdx.x];
        } else {
            local_max = -FLT_MAX;
            local_arg = -1;
        }
        
        if (threadIdx.x < WARP_SIZE) {
            mask = 0xffffffff;
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                float other_val = __shfl_down_sync(mask, local_max, offset);
                int other_arg = __shfl_down_sync(mask, local_arg, offset);
                if (other_val > local_max) {
                    local_max = other_val;
                    local_arg = other_arg;
                } else if (other_val == local_max && other_arg < local_arg) {
                    local_arg = other_arg;
                }
            }
            if (threadIdx.x == 0) {
                indices[idx] = local_arg;
            }
        }
        __syncthreads(); // Synchronize before proceeding to the next (outer, inner) pair
    }
}

// Host function to launch the CUDA kernel
// Computes the outer, reduction (dim) and inner sizes from the input tensor.
// The output tensor has the same shape as the input with the reduction dimension removed.
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    int outerSize = 1;
    for (int i = 0; i < dim; i++) {
        outerSize *= sizes[i];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int i = dim + 1; i < ndim; i++) {
        innerSize *= sizes[i];
    }

    // Build the output shape by removing the reduction dimension
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i != dim)
            out_sizes.push_back(sizes[i]);
    }

    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Launch configuration
    const int threads = 256;
    int total = outerSize * innerSize;
    int blocks = (total < 1024) ? total : 1024;  // Cap grid size to 1024 blocks
    int nWarps = (threads + WARP_SIZE - 1) / WARP_SIZE;
    size_t shared_mem_size = nWarps * (sizeof(float) + sizeof(int));

    argmax_stride_loop_kernel<<<blocks, threads, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (stride-loop with boundary handling)");
}
