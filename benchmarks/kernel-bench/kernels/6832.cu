#include <torch/extension.h>
#include <vector>
#include <cfloat>

__global__ void coalesced_memory_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize,
    const int outerSize) {

    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    const int block_offset = blockIdx.x * blockDim.x;
    
    const int global_idx = block_offset + tid;
    const int inner_group = global_idx % innerSize;
    const int outer_group = global_idx / innerSize;
    
    if (outer_group >= outerSize) return;

    float thread_max = -FLT_MAX;
    int thread_max_idx = 0;

    // Coalesced memory access pattern
    #pragma unroll 4
    for (int d = 0; d < dimSize; d++) {
        const int offset = outer_group * dimSize * innerSize + 
                          d * innerSize + 
                          inner_group;
        
        float val = __ldg(&x[offset]);
        if (val > thread_max) {
            thread_max = val;
            thread_max_idx = d;
        }
    }

    extern __shared__ float shared[];
    int* sidx = (int*)&shared[blockDim.x];

    shared[tid] = thread_max;
    sidx[tid] = thread_max_idx;
    __syncthreads();

    // Warp-level reduction
    if (tid < 32) {
        for (int offset = warp_id * warpSize; offset < (warp_id + 1) * warpSize && offset < blockDim.x; offset++) {
            if (shared[offset] > thread_max) {
                thread_max = shared[offset];
                thread_max_idx = sidx[offset];
            }
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
            if (other_max > thread_max) {
                thread_max = other_max;
                thread_max_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            shared[warp_id] = thread_max;
            sidx[warp_id] = thread_max_idx;
        }
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 1; i < blockDim.x/warpSize; i++) {
            if (shared[i] > thread_max) {
                thread_max = shared[i];
                thread_max_idx = sidx[i];
            }
        }
        indices[blockIdx.x] = thread_max_idx;
    }
}

torch::Tensor coalesced_memory_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
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

    const int blockSize = 128;
    const int numBlocks = (outerSize * innerSize + blockSize - 1) / blockSize;
    const size_t shared_mem_size = blockSize * (sizeof(float) + sizeof(int));

    coalesced_memory_argmax_kernel<<<numBlocks, blockSize, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize,
        outerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &coalesced_memory_argmax_forward_cuda, "Coalesced Memory ArgMax CUDA forward");
}