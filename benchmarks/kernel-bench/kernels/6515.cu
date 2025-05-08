#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute the mean along the reduction dimension
template <typename scalar_t>
__device__ inline scalar_t compute_mean(const scalar_t* input,
                                           int outer_idx,
                                           int inner_idx,
                                           int dim_size,
                                           int inner_size) {
    int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    scalar_t sum = static_cast<scalar_t>(0);
    for (int i = 0; i < dim_size; i++) {
        sum += input[input_offset + i * inner_size];
    }
    return sum / static_cast<scalar_t>(dim_size);
}

// Kernel using a grid-stride loop and the modular device function
template <typename scalar_t>
__global__ void mean_reduce_kernel(const scalar_t* input,
                                      scalar_t* output,
                                      int outer_size,
                                      int dim_size,
                                      int inner_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Each thread processes multiple elements using a grid-stride loop
    for (int idx = tid; idx < outer_size * inner_size; idx += stride) {
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;
        output[idx] = compute_mean<scalar_t>(input, outer_idx, inner_idx, dim_size, inner_size);
    }
}

// Host function launched from Python
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();
    
    // Obtain size info
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Create output tensor with the reduced dimension removed
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    // Launch kernel
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Modular mean reduction (CUDA)");
}
