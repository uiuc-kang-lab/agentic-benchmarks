#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Constants
const int BLOCK_SIZE = 256;

// Kernel definition
__global__ void optimized_cumsum_kernel(const float* input, float* output, int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        // Shared memory for block-level scan
        extern __shared__ float sdata[];
        float sum = 0.0f;

        // Each thread processes a chunk of the stride
        int tid = threadIdx.x;
        int numThreads = blockDim.x;
        int chunk_size = (stride + numThreads - 1) / numThreads;
        int start = tid * chunk_size;
        int end = min(start + chunk_size, stride);

        // Phase 1: Compute local chunk sum
        for (int i = start; i < end; ++i) {
            int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
            sum += input[idx];
            output[idx] = sum;
        }

        // Store local sum in shared memory
        sdata[tid] = sum;
        __syncthreads();

        // Perform block-level inclusive scan
        for (int offset = 1; offset < numThreads; offset *= 2) {
            float temp = 0.0f;
            if (tid >= offset) {
                temp = sdata[tid - offset];
            }
            __syncthreads();
            sdata[tid] += temp;
            __syncthreads();
        }

        // Convert to exclusive scan
        float thread_offset = (tid == 0) ? 0.0f : sdata[tid - 1];

        // Phase 2: Apply offset to cumulative sum
        sum = thread_offset;
        for (int i = start; i < end; ++i) {
            int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
            sum += input[idx];
            output[idx] = sum;
        }
    }
}

// Forward function
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

    // Use a 2D grid: one dimension for outer indices, one for covering the inner dimension based on BLOCK_SIZE.
    dim3 grid(outer_size, (inner_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel with shared memory allocation
    optimized_cumsum_kernel<<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum with combined techniques");
}