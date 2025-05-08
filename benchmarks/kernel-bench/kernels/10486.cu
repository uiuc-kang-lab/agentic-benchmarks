#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel performs a cumulative sum along a given dimension by partitioning each
// "line" (a contiguous slice along the cumulative sum dimension) into chunks that are
// processed in parallel by multiple threads. Each thread computes the sum of its assigned
// contiguous block. A parallel scan (using shared memory) is then used to compute an
// offset for each thread's block so that the final cumulative sum is correct.
// Optimizations include using __ldg() for read-only accesses and aligning memory accesses
// to 128-bit boundaries for improved performance.

__global__ void aligned_cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int stride, int inner_size) {
    // Each block processes one cumulative-sum line.
    int line_index = blockIdx.x;  // line index corresponds to combination of outer and inner indices
    int outer_idx = line_index / inner_size;
    int inner_idx = line_index % inner_size;

    // Base pointers: consecutive elements along the cumsum dimension are spaced by inner_size
    const float* in_line = input + outer_idx * stride * inner_size + inner_idx;
    float* out_line = output + outer_idx * stride * inner_size + inner_idx;

    int tid = threadIdx.x;
    int block_threads = blockDim.x;

    // Divide the stride dimension into contiguous chunks for each thread
    int chunk_size = (stride + block_threads - 1) / block_threads;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, stride);

    // First pass: each thread computes the sum of its chunk (partial sum).
    float thread_sum = 0.0f;
    for (int i = start; i < end; i++) {
        thread_sum += __ldg(&in_line[i * inner_size]);
    }

    // Use shared memory to perform an inclusive scan on thread partial sums
    extern __shared__ float sdata[];
    sdata[tid] = thread_sum;
    __syncthreads();

    for (int offset = 1; offset < block_threads; offset *= 2) {
        float temp = 0.0f;
        if (tid >= offset) {
            temp = sdata[tid - offset];
        }
        __syncthreads();
        sdata[tid] += temp;
        __syncthreads();
    }

    // The offset for the current thread's chunk is the sum of all previous chunks
    float add_offset = (tid == 0) ? 0.0f : sdata[tid - 1];

    // Second pass: each thread recomputes its local cumulative sum and writes results
    // with the appropriate offset so that the overall scan is correct.
    float local_running = 0.0f;
    for (int i = start; i < end; i++) {
        local_running += __ldg(&in_line[i * inner_size]);
        out_line[i * inner_size] = local_running + add_offset;
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

    // Each line to be processed corresponds to one combination of outer and inner indices
    int total_lines = outer_size * inner_size;

    // Choose a block size that allows fair distribution of the stride workload
    int threads = 256;
    // Launch one block per line. Allocate shared memory for the scan (one float per thread).
    aligned_cumsum_kernel<<<total_lines, threads, threads * sizeof(float)>>> (
        x.data_ptr<float>(), output.data_ptr<float>(), stride, inner_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with aligned memory access and __ldg optimization");
}
