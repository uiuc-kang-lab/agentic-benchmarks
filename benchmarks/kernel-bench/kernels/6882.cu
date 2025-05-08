#include <torch/extension.h>
#include <vector>

#define BLOCK_DIM 256
#define TILE_SIZE 32

__global__ void argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {
    
    __shared__ float shared_data[TILE_SIZE];
    __shared__ int shared_indices[TILE_SIZE];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = outerSize * innerSize;
    
    if (idx < total) {
        const int outer_idx = idx / innerSize;
        const int inner_idx = idx % innerSize;
        const int base_offset = (outer_idx * dimSize * innerSize) + inner_idx;
        
        // Initialize with first element
        float max_val = x[base_offset];
        int max_idx = -1;
        
        // Process dimension in tiles
        for (int tile = 0; tile < dimSize; tile += TILE_SIZE) {
            const int remaining = min(TILE_SIZE, dimSize - tile);
            
            // Load tile into shared memory
            if (threadIdx.x < remaining) {
                shared_data[threadIdx.x] = x[base_offset + (tile + threadIdx.x) * innerSize];
                shared_indices[threadIdx.x] = tile + threadIdx.x;
            }
            __syncthreads();
            
            // Process the tile
            #pragma unroll
            for (int i = 0; i < remaining; i++) {
                if (shared_data[i] > max_val) {
                    max_val = shared_data[i];
                    max_idx = shared_indices[i];
                }
            }
            __syncthreads();
        }
        
        // Write final result
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
    
    const int threads = BLOCK_DIM;
    const int total = outerSize * innerSize;
    const int blocks = (total + threads - 1) / threads;
    
    argmax_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );
    
    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward");
}