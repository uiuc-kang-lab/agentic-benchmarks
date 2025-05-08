#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses shared memory to store intermediate results of the cumulative sum
// to reduce global memory accesses and improve performance.

__global__ void shared_memory_cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int stride, int inner_size) {
    extern __shared__ float sdata[] __restrict__;

    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;
    const int line_index = blockIdx.x;
    const int outer_idx = line_index / inner_size;
    const int inner_idx = line_index % inner_size;

    const float* in_line = input + outer_idx * stride * inner_size + inner_idx;
    float* out_line = output + outer_idx * stride * inner_size + inner_idx;

    const int chunk_size = (stride + block_threads - 1) / block_threads;
    const int start = tid * chunk_size;
    const int end = min(start + chunk_size, stride);

    // First pass: compute local sum with fewer registers
    float thread_sum = 0.0f;
    #pragma unroll 4
    for (int i = start; i < end; i++) {
        thread_sum += __ldg(&in_line[i * inner_size]);
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();

    // Optimized parallel reduction in shared memory
    #pragma unroll
    for (int offset = block_threads/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    // Compute prefix sum offset
    const float add_offset = (tid == 0) ? 0.0f : sdata[tid - 1];
    __syncthreads();  // Ensure all threads have their offset

    // Second pass: compute final output with minimal register usage
    float running_sum = add_offset;
    #pragma unroll 4
    for (int i = start; i < end; i++) {
        running_sum += __ldg(&in_line[i * inner_size]);
        out_line[i * inner_size] = running_sum;
    }
}

// The forward function sets up the grid to cover each "line" of the tensor along the cumsum dimension
// and launches the kernel with a fixed number of threads per block.

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

    int total_lines = outer_size * inner_size;

    int threads = 256;
    shared_memory_cumsum_kernel<<<total_lines, threads, threads * sizeof(float)>>> (
        x.data_ptr<float>(), output.data_ptr<float>(), stride, inner_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with shared memory optimization");
}
