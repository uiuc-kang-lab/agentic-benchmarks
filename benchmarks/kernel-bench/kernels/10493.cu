#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Threshold below which a simple sequential cumulative sum is more efficient
const int SEQ_THRESHOLD = 32;

// Hybrid kernel: each block processes one "line" of data corresponding to a fixed
// outer and inner index. For small stride (cumsum dimension length), we use a sequential
// accumulation to avoid the overhead of synchronization, and for large strides we use an
// even distribution of work with an intra-block scan over the partial sums.

__global__ void cumsum_hybrid_kernel(const float* input, float* output, int stride, int inner_size) {
    // Each block processes one line (combination of outer and inner indices).
    // Total number of lines = outer_size * inner_size.
    int line_index = blockIdx.x;
    int outer_idx = line_index / inner_size;
    int inner_idx = line_index % inner_size;

    // Pointers to the beginning of the line in memory. Elements along the cumulative
    // dimension are separated by inner_size.
    const float* in_line = input + outer_idx * stride * inner_size + inner_idx;
    float* out_line = output + outer_idx * stride * inner_size + inner_idx;

    int tid = threadIdx.x;
    int numThreads = blockDim.x;

    // If only one thread is launched, or the stride is small, execute a simple sequential cumsum.
    if (numThreads == 1 || stride <= SEQ_THRESHOLD) {
        if (tid == 0) {
            float sum = 0.0f;
            for (int i = 0; i < stride; ++i) {
                sum += in_line[i * inner_size];
                out_line[i * inner_size] = sum;
            }
        }
    } else {
        // Evenly partition the stride among the threads in this block.
        int chunk_size = (stride + numThreads - 1) / numThreads;  // ceiling division
        int start = tid * chunk_size;
        int end = start + chunk_size;
        if (end > stride) end = stride;

        // Phase 1: Each thread computes the sum of its assigned chunk.
        float thread_total = 0.0f;
        for (int i = start; i < end; ++i) {
            thread_total += in_line[i * inner_size];
        }

        // Use shared memory to perform an inclusive scan over thread_total values.
        extern __shared__ float sdata[];
        sdata[tid] = thread_total;
        __syncthreads();

        for (int offset = 1; offset < numThreads; offset *= 2) {
            float temp = 0.0f;
            if (tid >= offset) {
                temp = sdata[tid - offset];
            }
            __syncthreads();
            sdata[tid] += temp;
            __syncthreads();
        }

        // Convert the inclusive scan to an exclusive scan by subtracting the thread's own input
        float thread_offset = (tid == 0) ? 0.0f : sdata[tid - 1];

        // Phase 2: Each thread recomputes its chunk's local cumulative sum and adds the offset.
        float running = thread_offset;
        for (int i = start; i < end; ++i) {
            running += in_line[i * inner_size];
            out_line[i * inner_size] = running;
        }
    }
}

// The forward function sets up the tensor dimensions, selects the number of threads based on the
// stride (cumulative dimension length), and launches one block per "line" (each unique combination
// of outer and inner indices).

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    // Compute outer_size: product of dimensions before the cumsum dimension
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= x.size(i);
    }

    // Compute inner_size: product of dimensions after the cumsum dimension
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= x.size(i);
    }

    // The length of the cumulative sum dimension
    int stride = x.size(dim);

    // Total number of lines to process
    int total_lines = outer_size * inner_size;

    int threads;
    size_t smem_size = 0;

    // For small strides, a sequential approach is preferred
    if (stride <= SEQ_THRESHOLD) {
        threads = 1;
    } else {
        // For larger strides, use parallel processing with a maximum of 256 threads per block
        threads = (stride < 256) ? stride : 256;
        smem_size = threads * sizeof(float);
    }

    // Launch one block per line
    cumsum_hybrid_kernel<<<total_lines, threads, smem_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), stride, inner_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid CUDA cumulative sum kernel with dynamic mode selection");
}
