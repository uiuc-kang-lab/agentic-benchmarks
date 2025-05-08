#include <torch/extension.h>
#include <vector>
#include <cfloat>

__global__ void dimension_aware_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    // 2D grid: x-dimension handles outer indices, y-dimension handles inner indices
    const int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int inner_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (outer_idx >= outerSize || inner_idx >= innerSize) return;

    // Calculate base offset for this (outer, inner) pair
    const int base_offset = outer_idx * dimSize * innerSize + inner_idx;
    
    // Local maximum tracking
    float local_max = -FLT_MAX;
    int local_max_idx = 0;

    // Grid-stride loop over dimension
    #pragma unroll 4
    for (int dim_idx = 0; dim_idx < dimSize; dim_idx++) {
        float val = __ldg(&x[base_offset + dim_idx * innerSize]);
        if (val > local_max) {
            local_max = val;
            local_max_idx = dim_idx;
        }
    }

    // Shared memory for reduction within block
    extern __shared__ float shared[];
    int* sidx = (int*)&shared[blockDim.x * blockDim.y];
    
    // Linear thread index within the block
    const int tid = threadIdx.y + threadIdx.x * blockDim.y;
    const int block_size = blockDim.x * blockDim.y;
    
    // Store in shared memory
    shared[tid] = local_max;
    sidx[tid] = local_max_idx;
    __syncthreads();

    // Reduction within block
    for (int s = block_size/2; s > 32; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) {
                shared[tid] = shared[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction (last 32 elements)
    if (tid < 32) {
        volatile float* vshared = shared;
        volatile int* vsidx = sidx;
        if (block_size >= 64) {
            if (vshared[tid + 32] > vshared[tid]) {
                vshared[tid] = vshared[tid + 32];
                vsidx[tid] = vsidx[tid + 32];
            }
        }
        if (block_size >= 32) {
            if (vshared[tid + 16] > vshared[tid]) {
                vshared[tid] = vshared[tid + 16];
                vsidx[tid] = vsidx[tid + 16];
            }
        }
        if (block_size >= 16) {
            if (vshared[tid + 8] > vshared[tid]) {
                vshared[tid] = vshared[tid + 8];
                vsidx[tid] = vsidx[tid + 8];
            }
        }
        if (block_size >= 8) {
            if (vshared[tid + 4] > vshared[tid]) {
                vshared[tid] = vshared[tid + 4];
                vsidx[tid] = vsidx[tid + 4];
            }
        }
        if (block_size >= 4) {
            if (vshared[tid + 2] > vshared[tid]) {
                vshared[tid] = vshared[tid + 2];
                vsidx[tid] = vsidx[tid + 2];
            }
        }
        if (block_size >= 2) {
            if (vshared[tid + 1] > vshared[tid]) {
                vshared[tid] = vshared[tid + 1];
                vsidx[tid] = vsidx[tid + 1];
            }
        }
    }

    // Write result
    if (tid == 0) {
        indices[outer_idx * innerSize + inner_idx] = sidx[0];
    }
}

torch::Tensor dimension_aware_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 supported");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim");

    int outerSize = 1;
    for (int d = 0; d < dim; d++) outerSize *= sizes[d];
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) innerSize *= sizes[d];

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d != dim) out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // 2D block configuration
    dim3 block(16, 8);  // 128 threads total
    dim3 grid(
        (outerSize + block.x - 1) / block.x,
        (innerSize + block.y - 1) / block.y
    );

    // Shared memory size calculation
    const int block_size = block.x * block.y;
    const size_t shared_mem_size = block_size * (sizeof(float) + sizeof(int));

    dimension_aware_argmax_kernel<<<grid, block, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dimension_aware_argmax_forward_cuda, "Dimension-Aware ArgMax CUDA forward");
}