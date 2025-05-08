#include <torch/extension.h>
#include <vector>

// Device function to compute argmax over a slice along the specified dimension
// This modular function improves code readability and ease of future modifications.

// __forceinline__ hints to inline the function for performance
__device__ __forceinline__ int argmax_slice(const float* __restrict__ x, 
                                               const int offset, 
                                               const int dimSize, 
                                               const int stride) {
    float max_val = x[offset];
    int max_idx = 0;
    for (int d = 1; d < dimSize; ++d) {
        float current = x[offset + d * stride];
        if (current > max_val) {
            max_val = current;
            max_idx = d;
        }
    }
    return max_idx;
}

// CUDA kernel for argmax over a specified dimension using the modular device function
// Each thread processes one (outer, inner) pair
__global__ void argmax_kernel_modular(const float* __restrict__ x,
                                        int64_t* __restrict__ indices,
                                        const int outerSize,
                                        const int dimSize,
                                        const int innerSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;
    if (idx < total) {
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;
        int offset = outer_idx * dimSize * innerSize + inner_idx;
        
        // Use the modular device function to compute the argmax in the slice
        int max_idx = argmax_slice(x, offset, dimSize, innerSize);
        indices[outer_idx * innerSize + inner_idx] = max_idx;
    }
}

// Host function to launch the modular CUDA kernel
// It calculates necessary sizes, prepares output tensor, and launches the kernel
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

    // Output tensor has the input shape with the specified dimension removed
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

    argmax_kernel_modular<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward using modular device functions");
}
