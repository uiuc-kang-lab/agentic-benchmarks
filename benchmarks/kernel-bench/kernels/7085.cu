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
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int outer_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;
    
    if (outer_idx >= outer || inner_idx >= inner) return;
    
    // Starting index for reduction in the r dimension
    const int base = outer_idx * (r * inner) + inner_idx;
    scalar_t min_val = input[base];
    
    // First pass: each thread finds its local minimum
    #pragma unroll
    for (int j = 1; j < r; j++) {
        const int index = base + j * inner;
        const scalar_t curr = input[index];
        min_val = curr < min_val ? curr : min_val;
    }
    
    // Store in shared memory
    shared_data[tid] = min_val;
    __syncthreads();
    
    // Block-level reduction in shared memory
    for (int s = block_size/2; s > 0; s >>= 1) {
        if (tid < s) {
            scalar_t other = shared_data[tid + s];
            scalar_t mine = shared_data[tid];
            shared_data[tid] = other < mine ? other : mine;
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[outer_idx * inner + inner_idx] = shared_data[0];
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
    
    // Use 2D grid and block configuration
    dim3 threads(16, 16);
    dim3 blocks(
        (inner + threads.x - 1) / threads.x,
        (outer + threads.y - 1) / threads.y
    );
    
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