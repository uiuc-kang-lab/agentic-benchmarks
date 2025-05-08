#include <torch/extension.h>
#include <vector>

// Dynamic block size computation with architecture-aware optimization
__host__ int compute_optimal_block_size(int total_elements, int dim_size) {
    // Consider both total elements and dimension size for optimization
    if (total_elements < 1024 || dim_size < 32) {
        return 32;
    } else if (total_elements < 4096 || dim_size < 64) {
        return 64;
    } else if (total_elements < 16384 || dim_size < 128) {
        return 128;
    } else if (total_elements < 65536 || dim_size < 256) {
        return 256;
    } else {
        return 512;
    }
}

template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void adaptive_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    // Use shared memory for frequently accessed data
    __shared__ float shared_max[32];  // For block reduction if needed
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;

    if (idx < total) {
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;
        int start_offset = outer_idx * dimSize * innerSize + inner_idx;

        // Register-based computation for better performance
        float max_val = x[start_offset];
        int max_idx = 0;

        // Unrolled loop for better instruction-level parallelism
        #pragma unroll 4
        for (int d = 1; d < dimSize; d++) {
            float val = x[start_offset + d * innerSize];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }

        // Write result directly to global memory
        indices[outer_idx * innerSize + inner_idx] = max_idx;
    }
}

torch::Tensor adaptive_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported");
    auto x_contig = x.contiguous();
    
    auto sizes = x_contig.sizes();
    auto ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax");

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

    int total = outerSize * innerSize;
    int block_size = compute_optimal_block_size(total, dimSize);
    int blocks = (total + block_size - 1) / block_size;

    // Template-based dispatch for different block sizes
    switch(block_size) {
        case 32:
            adaptive_argmax_kernel<32><<<blocks, 32>>>(
                x_contig.data_ptr<float>(), indices.data_ptr<int64_t>(),
                outerSize, dimSize, innerSize);
            break;
        case 64:
            adaptive_argmax_kernel<64><<<blocks, 64>>>(
                x_contig.data_ptr<float>(), indices.data_ptr<int64_t>(),
                outerSize, dimSize, innerSize);
            break;
        case 128:
            adaptive_argmax_kernel<128><<<blocks, 128>>>(
                x_contig.data_ptr<float>(), indices.data_ptr<int64_t>(),
                outerSize, dimSize, innerSize);
            break;
        case 256:
            adaptive_argmax_kernel<256><<<blocks, 256>>>(
                x_contig.data_ptr<float>(), indices.data_ptr<int64_t>(),
                outerSize, dimSize, innerSize);
            break;
        default:
            adaptive_argmax_kernel<512><<<blocks, 512>>>(
                x_contig.data_ptr<float>(), indices.data_ptr<int64_t>(),
                outerSize, dimSize, innerSize);
    }
    
    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &adaptive_argmax_forward_cuda, "Adaptive Optimized ArgMax CUDA forward");
}