#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
    
    __shared__ scalar_t shared_data[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total) return;

    int outer_idx = idx / inner;
    int inner_idx = idx % inner;
    int base = outer_idx * (r * inner) + inner_idx;
    
    // Load first element and initialize min_val
    scalar_t min_val = input[base];
    
    // Each thread processes its portion of the reduction dimension
    #pragma unroll 4
    for (int j = 1; j < r; j++) {
        scalar_t curr = input[base + j * inner];
        min_val = curr < min_val ? curr : min_val;
    }
    
    // Store thread's local minimum in shared memory
    shared_data[threadIdx.x] = min_val;\n    __syncthreads();\n    // Perform reduction within thread block\n    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {\n        if (threadIdx.x < offset) {\n            shared_data[threadIdx.x] = min(shared_data[threadIdx.x], shared_data[threadIdx.x + offset]);\n        }\n        __syncthreads();\n    }\n    if (threadIdx.x == 0) {\n        output[blockIdx.x] = shared_data[0];\n    }
    __syncthreads();
    
    // Perform reduction within thread block if needed
    if (threadIdx.x == 0) {
        scalar_t block_min = shared_data[0];
        #pragma unroll
        for (int i = 1; i < blockDim.x && (blockIdx.x * blockDim.x + i < total); i++) {
            block_min = min(block_min, shared_data[i]);
        }
        output[idx] = block_min;
    }
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int ndim = input.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    int outer = 1;
    for (int i = 0; i < dim; i++) {
        outer *= input.size(i);
    }
    int r = input.size(dim);
    int inner = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner *= input.size(i);
    }

    std::vector<int64_t> output_shape;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            output_shape.push_back(input.size(i));
        }
    }
    
    auto output = torch::empty(output_shape, input.options());

    int total = outer * inner;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_cuda", ([&] {
        min_reduce_kernel<scalar_t><<<blocks, threads, 0, 
            c10::cuda::getCurrentCUDAStream().stream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer,
            r,
            inner);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Min reduction over a specified dimension (CUDA)");
}