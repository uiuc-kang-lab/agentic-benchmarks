#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Modular device function to compute the partial sum for a given range [start, end) along the cumsum dimension.
__device__ float compute_partial_sum(const float* in_line, int start, int end, int inner_size) {
    float sum = 0.0f;
    for (int i = start; i < end; ++i) {
        sum += in_line[i * inner_size];
    }
    return sum;
}

// Modular device function to perform an inclusive scan in shared memory.
// sdata: pointer to shared memory; tid: thread index; n: total threads in the block.
__device__ void inclusive_scan(volatile float* sdata, int tid, int n) {
    for (int offset = 1; offset < n; offset *= 2) {
        float temp = (tid >= offset) ? sdata[tid - offset] : 0.0f;
        __syncthreads();
        sdata[tid] += temp;
        __syncthreads();
    }
}

// Modular device function to compute the local cumulative sum for a thread's chunk and write outputs.
__device__ void compute_local_cumsum(const float* in_line, float* out_line, int start, int end, int inner_size, float add_offset) {
    float local_sum = 0.0f;
    for (int i = start; i < end; ++i) {
        local_sum += in_line[i * inner_size];
        out_line[i * inner_size] = local_sum + add_offset;
    }
}

// Kernel: Each block processes one "line" (a slice along the cumulative sum dimension defined by outer and inner indices).
// The work along the cumulative dimension (of length 'stride') is partitioned among threads.
__global__ void modular_cumsum_kernel(const float* input, float* output, int stride, int inner_size) {
    // Determine the line being processed
    int line = blockIdx.x;  // Each block processes one line
    int outer = line / inner_size;
    int inner = line % inner_size;

    // Base pointers. Consecutive elements along the cumulative dimension are spaced by inner_size.
    const float* in_line = input + outer * stride * inner_size + inner;
    float* out_line = output + outer * stride * inner_size + inner;

    int tid = threadIdx.x;
    int numThreads = blockDim.x;

    // Partition the cumulative (stride) dimension among threads using contiguous chunks.
    int chunk_size = (stride + numThreads - 1) / numThreads;
    int start = tid * chunk_size;
    int end = start + chunk_size;
    if (end > stride) end = stride;

    // Phase 1: Each thread computes the partial sum for its chunk using a modular device function.
    float thread_sum = compute_partial_sum(in_line, start, end, inner_size);

    // Use shared memory to perform an inclusive scan across thread partial sums.
    extern __shared__ float sdata[];
    sdata[tid] = thread_sum;
    __syncthreads();

    inclusive_scan(sdata, tid, numThreads);

    // Convert inclusive scan result into an exclusive offset for this thread.
    float offset = (tid == 0) ? 0.0f : sdata[tid - 1];
    __syncthreads();

    // Phase 2: Recompute local cumulative sum for each thread's chunk and add the computed offset
    compute_local_cumsum(in_line, out_line, start, end, inner_size, offset);
}

// The forward function prepares tensor dimensions, sets up grid and block sizes, and launches the kernel.
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

    // Choose an appropriate block size for distributing the work along the cumulative dimension.
    int threads = 256;
    // Launch one block per line and allocate shared memory (one float per thread).
    modular_cumsum_kernel<<<total_lines, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), stride, inner_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular CUDA cumulative sum via device functions");
}
