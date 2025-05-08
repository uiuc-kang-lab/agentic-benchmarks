#include <torch/extension.h>
#include <vector>
#include <float.h>

// Constant memory for frequently accessed dimensions
__constant__ int c_dimSize;

__global__ void hybrid_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int innerSize,
    const bool use_cooperative) {
    
    if (use_cooperative) {
        // Cooperative approach for large reduction dimensions
        int slice = blockIdx.x;
        if (slice >= outerSize * innerSize) return;

        int outer_idx = slice / innerSize;
        int inner_idx = slice % innerSize;
        int base_offset = outer_idx * (c_dimSize * innerSize) + inner_idx;

        // Grid-stride loop for coalesced memory access
        float local_max = -FLT_MAX;
        int local_argmax = 0;
        
        #pragma unroll 4
        for (int d = threadIdx.x; d < c_dimSize; d += blockDim.x) {
            float val = x[base_offset + d * innerSize];
            if (val > local_max) {
                local_max = val;
                local_argmax = d;
            }
        }

        extern __shared__ char shm[];
        float* s_max = reinterpret_cast<float*>(shm);
        int* s_idx = reinterpret_cast<int*>(s_max + blockDim.x);

        s_max[threadIdx.x] = local_max;
        s_idx[threadIdx.x] = local_argmax;
        __syncthreads();

        // Warp-level reduction first
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, local_argmax, offset);
            if (other_max > local_max) {
                local_max = other_max;
                local_argmax = other_idx;
            }
        }

        // Block-level reduction
        if (threadIdx.x % warpSize == 0) {
            s_max[threadIdx.x/warpSize] = local_max;
            s_idx[threadIdx.x/warpSize] = local_argmax;
        }
        __syncthreads();

        // Final reduction by first thread
        if (threadIdx.x == 0) {
            for (int i = 1; i < blockDim.x/warpSize; i++) {
                if (s_max[i] > s_max[0]) {
                    s_max[0] = s_max[i];
                    s_idx[0] = s_idx[i];
                }
            }
            indices[slice] = s_idx[0];
        }
    } else {
        // Simple approach for small reduction dimensions
        int outer_idx = blockIdx.y * blockDim.y + threadIdx.y;
        int inner_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (outer_idx < outerSize && inner_idx < innerSize) {
            int start_offset = outer_idx * (c_dimSize * innerSize) + inner_idx;
            float max_val = x[start_offset];
            int max_idx = 0;

            #pragma unroll 16
            for (int d = 1; d < c_dimSize; d++) {
                float val = x[start_offset + d * innerSize];
                if (val > max_val) {
                    max_val = val;
                    max_idx = d;
                }
            }

            indices[outer_idx * innerSize + inner_idx] = max_idx;
        }
    }
}

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported");
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax");

    int outerSize = 1;
    for (int d = 0; d < dim; d++) outerSize *= sizes[d];
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) innerSize *= sizes[d];

    cudaMemcpyToSymbol(c_dimSize, &dimSize, sizeof(int));

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d != dim) out_sizes.push_back(sizes[d]);
    }
    auto indices = torch::empty(out_sizes, 
        torch::TensorOptions().device(x.device()).dtype(torch::kLong));

    // Choose kernel configuration based on reduction size
    const bool use_cooperative = (dimSize >= 128);
    
    if (use_cooperative) {
        // For large reduction dimensions, use cooperative approach
        int block_size = min(256, dimSize);
        dim3 grid(outerSize * innerSize);
        dim3 block(block_size);
        size_t shm_size = block_size * (sizeof(float) + sizeof(int));
        
        hybrid_argmax_kernel<<<grid, block, shm_size>>>(
            x_contig.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            outerSize,
            innerSize,
            true
        );
    } else {
        // For small reduction dimensions, use simple approach
        dim3 block(32, 8);
        dim3 grid((innerSize + block.x - 1) / block.x,
                  (outerSize + block.y - 1) / block.y);
        
        hybrid_argmax_kernel<<<grid, block, 0>>>(
            x_contig.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            outerSize,
            innerSize,
            false
        );
    }

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "Hybrid adaptive ArgMax CUDA forward");
}