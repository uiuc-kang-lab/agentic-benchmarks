#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel performs the cumulative sum along a given dimension by splitting each
// "line" (defined by outer and inner indices) into chunks. Loop unrolling via
// #pragma unroll is applied in critical loops to reduce overhead and improve performance.

__global__ void parallel_cumsum_kernel_unroll(const float* input, float* output, int stride, int inner_size) {
    // Each block processes one cumulative-sum line.
    int line_index = blockIdx.x;  // line index corresponds to one combination of outer and inner indices
    int outer_idx = line_index / inner_size;
    int inner_idx = line_index % inner_size;

    // Base pointer for this cumsum line. Elements along the stride are spaced by inner_size.
    const float* in_line = input + outer_idx * stride * inner_size + inner_idx;
    float* out_line = output + outer_idx * stride * inner_size + inner_idx;

    int tid = threadIdx.x;
    int block_threads = blockDim.x;

    // Determine each thread's chunk boundaries
    int chunk_size = (stride + block_threads - 1) / block_threads;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, stride);

    // First pass: each thread computes the partial sum of its chunk
    float thread_sum = 0.0f;
    #pragma unroll
    for (int i = start; i < end; i++) {
        thread_sum += in_line[i * inner_size];
    }

    // Use shared memory for an inclusive scan of the partial sums
    extern __shared__ float sdata[];
    sdata[tid] = thread_sum;
    __syncthreads();

    #pragma unroll
    for (int offset = 1; offset < block_threads; offset *= 2) {
        float temp = 0.0f;
        if (tid >= offset) {
            temp = sdata[tid - offset];
        }
        __syncthreads();
        sdata[tid] += temp;
        __syncthreads();
    }

    // Offset for the current thread's chunk is the sum of all previous chunks
    float add_offset = (tid == 0) ? 0.0f : sdata[tid - 1];

    // Second pass: each thread computes the cumulative sum for its chunk and writes results
    float local_running = 0.0f;
    #pragma unroll
    for (int i = start; i < end; i++) {
        local_running += in_line[i * inner_size];
        out_line[i * inner_size] = local_running + add_offset;
    }
}

// The forward function sets up the grid and launches the kernel. Each block processes one cumsum line
// (a combination of outer and inner indices). Loop unrolling in the kernel reduces loop overhead on the H100 GPU.

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

    // Choose a block size that is likely to cover a reasonable portion of the stride dimension.
    int threads = 256;

    // Launch one block per cumsum line and allocate shared memory for the reduction (one float per thread).
    parallel_cumsum_kernel_unroll<<<total_lines, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), stride, inner_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with loop unrolling optimizations");
}
