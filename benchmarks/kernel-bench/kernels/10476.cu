#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel performs a cumulative sum along a given dimension for a single "line" of data.
// A "line" corresponds to a fixed combination of outer and inner indices, and the cumulative
// dimension is of length 'stride'.
// We evenly partition the stride among the threads in the block by assigning each thread a contiguous
// chunk. In Phase 1, each thread computes the sum of its chunk. Then, using shared memory, an exclusive
// scan is performed over these thread totals to obtain an offset for each thread. In Phase 2, each thread
// recomputes its local cumulative sum and adds the offset so that the final global cumulative sum is correct.

__global__ void cumsum_even_dist_kernel(const float* input, float* output, int stride, int inner_size) {
    // Each block processes one line (combination of outer and inner indices)
    int line_index = blockIdx.x;  // total number of lines = outer_size * inner_size
    int outer_idx = line_index / inner_size;
    int inner_idx = line_index % inner_size;

    // Pointers to the beginning of the line in input and output. Note: elements along the cumulative sum
    // dimension are spaced by 'inner_size'.
    const float* in_line = input + outer_idx * stride * inner_size + inner_idx;
    float* out_line = output + outer_idx * stride * inner_size + inner_idx;

    int tid = threadIdx.x;
    int numThreads = blockDim.x;

    // Partition the stride (cumsum dimension length) contiguously among threads.
    // Compute chunk size per thread.
    int chunk_size = (stride + numThreads - 1) / numThreads;  // ceiling division
    int start = tid * chunk_size;
    int end = start + chunk_size;
    if (end > stride) end = stride;

    // Phase 1: Each thread computes the sum of its chunk.
    float thread_total = 0.0f;
    for (int i = start; i < end; ++i) {
        // Access the i-th element along the cumulative dimension.
        float val = in_line[i * inner_size];
        thread_total += val;
    }

    // Use shared memory to compute an exclusive scan of thread_total across the block
    extern __shared__ float sdata[];  // size = numThreads * sizeof(float)
    sdata[tid] = thread_total;
    __syncthreads();

    // Perform an inclusive scan in shared memory.
    for (int offset = 1; offset < numThreads; offset *= 2) {
        float temp = 0.0f;
        if (tid >= offset) {
            temp = sdata[tid - offset];
        }
        __syncthreads();
        sdata[tid] += temp;
        __syncthreads();
    }

    // Convert inclusive scan to exclusive scan: thread_offset is sum of all totals from threads with id < tid.
    float thread_offset = (tid == 0) ? 0.0f : sdata[tid - 1];

    // Phase 2: Each thread recomputes the cumulative sum for its chunk and adds the offset
    float running = thread_offset;
    for (int i = start; i < end; ++i) {
        running += in_line[i * inner_size];
        out_line[i * inner_size] = running;
    }
}

// The forward function prepares the tensor dimensions and launches the kernel.
// We process each line (combination of outer and inner indices) with one block.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    auto output = torch::empty_like(x);

    int ndim = x.dim();
    // Normalize the cumulative sum dimension.
    dim = (dim + ndim) % ndim;

    // Compute outer and inner sizes relative to the cumulative dimension.
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= x.size(i);
    }
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= x.size(i);
    }
    int stride = x.size(dim);

    // Total number of lines to process.
    int total_lines = outer_size * inner_size;

    // Decide on a reasonable number of threads per block.
    // For best load balance, use min(256, stride) so that threads aren't underutilized when stride is small.
    int threads = (stride < 256) ? stride : 256;

    // Launch one block per line. Allocate shared memory for the scan: threads * sizeof(float).
    cumsum_even_dist_kernel<<<total_lines, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), stride, inner_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with even work distribution");
}
