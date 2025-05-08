#include <torch/extension.h>
#include <vector>

// CUDA kernel using warp-level primitives for argmax
__global__ void argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize)
{
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    
    if (idx < total) {
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;
        int start_offset = (outer_idx * dimSize * innerSize) + inner_idx;
        
        // Initialize max value and index
        float max_val = x[start_offset];
        int max_idx = 0;
        
        // Each thread processes its portion of the dimension
        for (int d = threadIdx.x % 32; d < dimSize; d += 32) {
            float val = x[start_offset + d * innerSize];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }
        
        // Warp reduction using shuffle operations
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            
            if (threadIdx.x % 32 < offset && other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }
        
        // First thread in each warp writes the result
        if (threadIdx.x % 32 == 0) {
            indices[outer_idx * innerSize + inner_idx] = max_idx;
        }
    }
}

// Host function to launch the CUDA kernel
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
    
    const int threads = 256;
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