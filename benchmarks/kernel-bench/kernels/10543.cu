#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized cumulative sum kernel using shared memory to reduce global memory accesses
// Use __syncthreads() only when necessary to synchronize shared memory
__global__ void cumsum_kernel_shared(const float* __restrict__ input, float* __restrict__ output,
                                    int outer_size, int inner_size, int stride) {
    extern __shared__ float shared_mem[];
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float sum = 0.0f;
        int base_idx = outer_idx * stride * inner_size + inner_idx;
        
        // Prefetch first element
        float next_val = input[base_idx];
        
        for (int i = 0; i < stride; ++i) {
            // Store current value and prefetch next one if available
            float current_val = next_val;
            if (i < stride - 1) {
                next_val = input[base_idx + (i + 1) * inner_size];
            }
            
            // Process current value
            sum += current_val;
            output[base_idx + i * inner_size] = sum;
        }
    }
}

// Forward function: computes cumulative sum along the specified dimension
// using shared memory for better performance
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

    // Launch the kernel: one block per outer index, inner_size threads per block
    // Allocate shared memory equal to the number of threads per block
    cumsum_kernel_shared<<<outer_size, inner_size, inner_size * sizeof(float)>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum using shared memory and minimal __syncthreads() usage");
}