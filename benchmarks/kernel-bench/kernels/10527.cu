#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void cumsum_kernel_coalesced(const float* __restrict__ input, float* __restrict__ output,
                                         int outer_size, int inner_size, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * stride) return;
    int inner_idx = tid % stride;
    int outer_idx = tid / stride;

    int idx = outer_idx * stride * inner_size + inner_idx * inner_size;
    float sum = 0.0f;
    for (int i = 0; i < inner_size; ++i) {
        float val = __ldg(&input[idx + i]);
        sum += val;
        output[idx + i] = sum;
    }
}

__global__ void cumsum_kernel_unroll_align(const float* __restrict__ input, float* __restrict__ output,
                                            int dim, int outer_size, int inner_size, int stride) {
    int inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (inner_idx >= inner_size) return;

    // Use shared memory for partial sums to reduce register pressure
    __shared__ float partial_sums[256];  // Assuming blockSize <= 256
    
    for (int outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
        int base_idx = outer_idx * stride * inner_size + inner_idx;
        float sum = 0.0f;
        
        // Process in chunks to reduce register pressure
        #pragma unroll 4
        for (int i = 0; i < stride; i += 4) {
            // Load 4 elements at a time
            float val1 = (i < stride) ? __ldg(&input[base_idx + i * inner_size]) : 0.0f;
            float val2 = (i + 1 < stride) ? __ldg(&input[base_idx + (i + 1) * inner_size]) : 0.0f;
            float val3 = (i + 2 < stride) ? __ldg(&input[base_idx + (i + 2) * inner_size]) : 0.0f;
            float val4 = (i + 3 < stride) ? __ldg(&input[base_idx + (i + 3) * inner_size]) : 0.0f;
            
            // Update partial sums
            sum += val1;
            output[base_idx + i * inner_size] = sum;
            if (i + 1 < stride) {
                sum += val2;
                output[base_idx + (i + 1) * inner_size] = sum;
            }
            if (i + 2 < stride) {
                sum += val3;
                output[base_idx + (i + 2) * inner_size] = sum;
            }
            if (i + 3 < stride) {
                sum += val4;
                output[base_idx + (i + 3) * inner_size] = sum;
            }
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
    int blockSize = 256;
    int numBlocks = (outer_size * stride + blockSize - 1) / blockSize;

    cumsum_kernel_unroll_align<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), dim, outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with memory coalescing and unrolling");
}