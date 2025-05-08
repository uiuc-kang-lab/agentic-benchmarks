#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses shared memory to store intermediate results of the cumulative sum
// to reduce global memory accesses and improve performance.

__global__ void shared_memory_cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int stride, int inner_size) {
    extern __shared__ float sdata[];

    int line_index = blockIdx.x;
    int outer_idx = line_index / inner_size;
    int inner_idx = line_index % inner_size;

    const float* in_line = input + outer_idx * stride * inner_size + inner_idx;
    float* out_line = output + outer_idx * stride * inner_size + inner_idx;

    int tid = threadIdx.x;
    int block_threads = blockDim.x;

    int chunk_size = (stride + block_threads - 1) / block_threads;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, stride);

    float thread_sum = 0.0f;
    for (int i = start; i < end; i++) {
        thread_sum += __ldg(&in_line[i * inner_size]);
    }

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

    float add_offset = (tid == 0) ? 0.0f : sdata[tid - 1];

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
