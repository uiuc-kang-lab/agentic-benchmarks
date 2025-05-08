#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void cumsum_kernel_stride(const float* input, float* output, int stride, int inner_size, int total_size) {
    int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outer_idx < total_size) {
        int base_idx = (outer_idx / inner_size) * stride * inner_size + (outer_idx % inner_size);
        
        // Use shared memory for partial sums to reduce global memory accesses
        __shared__ float partial_sums[256];  // Assuming block size is 256
        float sum = 0.0f;
        
        // Process elements in chunks to maximize memory coalescing
        #pragma unroll 4
        for (int i = 0; i < stride; ++i) {
            int current_idx = base_idx + i * inner_size;
            sum += input[current_idx];
            output[current_idx] = sum;
            
            // Store partial sum in shared memory for potential reuse
            partial_sums[threadIdx.x] = sum;
            __syncthreads();
        }
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= x.size(i);
    }

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= x.size(i);
    }

    int stride = x.size(dim);
    int total_size = outer_size * inner_size;

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;

    cumsum_kernel_stride<<<blocks, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), stride, inner_size, total_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with stride handling");
}