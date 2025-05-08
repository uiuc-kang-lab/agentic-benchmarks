#include <torch/extension.h>
#include <vector>

__global__ void argmax_kernel_aligned(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize)
{
    // Align to warp size (32)
    const int tid = threadIdx.x;
    const int wid = threadIdx.y;
    const int bid = blockIdx.x;
    
    // Calculate global position
    const int warp_size = 32;
    const int warps_per_block = blockDim.y;
    const int global_warp_id = bid * warps_per_block + wid;
    const int global_idx = global_warp_id * warp_size + tid;
    
    if (global_idx < outerSize * innerSize) {
        const int outer_idx = global_idx / innerSize;
        const int inner_idx = global_idx % innerSize;
        
        // Calculate base offset for this thread
        const int base_offset = outer_idx * dimSize * innerSize + inner_idx;
        
        // Initialize max value and index
        float max_val = x[base_offset];
        int max_idx = 0;
        
        // Unroll first iterations to reduce branch divergence
        #pragma unroll 4
        for (int d = 0; d < (dimSize & ~3); d += 4) {
            float val1 = x[base_offset + (d + 1) * innerSize];
            float val2 = x[base_offset + (d + 2) * innerSize];
            float val3 = x[base_offset + (d + 3) * innerSize];
            float val4 = (d + 4 < dimSize) ? x[base_offset + (d + 4) * innerSize] : -INFINITY;
            
            // Use branchless comparisons
            bool cmp1 = val1 > max_val;
            bool cmp2 = val2 > max_val;
            bool cmp3 = val3 > max_val;
            bool cmp4 = val4 > max_val;
            
            max_val = cmp1 ? val1 : max_val;
            max_idx = cmp1 ? (d + 1) : max_idx;
            
            max_val = cmp2 ? val2 : max_val;
            max_idx = cmp2 ? (d + 2) : max_idx;
            
            max_val = cmp3 ? val3 : max_val;
            max_idx = cmp3 ? (d + 3) : max_idx;
            
            max_val = cmp4 ? val4 : max_val;
            max_idx = cmp4 ? (d + 4) : max_idx;
        }
        
        // Handle remaining elements
        for (int d = (dimSize & ~3); d < dimSize; d++) {
            float val = x[base_offset + d * innerSize];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }
        
        // Write result
        indices[outer_idx * innerSize + inner_idx] = max_idx;
    }
}

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    
    auto sizes = x_contig.sizes();
    auto ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");
    
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }
    
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    
    auto options = torch::TensorOptions()
                      .device(x.device())
                      .dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);
    
    // Configure grid and blocks for warp-aligned execution
    const int warp_size = 32;
    const int warps_per_block = 8;
    const int threads_per_block = warp_size * warps_per_block;
    const int total_elements = outerSize * innerSize;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    dim3 block(warp_size, warps_per_block);
    dim3 grid(num_blocks);
    
    argmax_kernel_aligned<<<grid, block>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );
    
    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (warp-aligned)");
}