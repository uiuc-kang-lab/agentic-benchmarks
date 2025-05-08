#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<int BLOCK_SIZE>
__global__ void cumsum_kernel_optimized(const float* __restrict__ input, 
                                      float* output,
                                      const int outer_size, 
                                      const int inner_size, 
                                      const int stride) {
    const int outer_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (outer_idx >= outer_size) return;

    // Shared memory for block-level reduction
    __shared__ float shared_data[BLOCK_SIZE];
    
    const int items_per_thread = (inner_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int base_outer = outer_idx * stride * inner_size;

    // Process multiple elements per thread using grid-stride loop
    for (int s = 0; s < stride; ++s) {
        const int stride_offset = s * inner_size;
        
        // Each thread processes its assigned elements
        float local_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < items_per_thread; ++i) {
            const int inner_idx = tid + i * BLOCK_SIZE;
            if (inner_idx < inner_size) {
                const int idx = base_outer + stride_offset + inner_idx;
                local_sum += __ldg(&input[idx]);
                output[idx] = local_sum;
            }
        }

        // Ensure all threads complete their local sums
        __syncthreads();
        
        // Share results for next iteration
        if (tid < inner_size) {
            shared_data[tid] = local_sum;
        }
        __syncthreads();
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
    
    constexpr int BLOCK_SIZE = 256;  // Optimal block size for most GPU architectures
    const dim3 blocks(outer_size);
    const dim3 threads(BLOCK_SIZE);

    cumsum_kernel_optimized<BLOCK_SIZE><<<blocks, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), 
        outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum");
}