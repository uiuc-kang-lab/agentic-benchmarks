#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel leverages shared memory to perform a parallel inclusive scan (cumulative sum) along the
// cumulative dimension (stride). Each block processes one (outer, inner) pair by loading the entire
// column (of length 'stride') into shared memory. A work-efficient Blelloch scan is performed in shared
// memory which reduces the number of global memory accesses and minimizes latency through parallelization.
// The resulting exclusive scan is converted to an inclusive scan by adding the original value.

__global__ void cumsum_shared_kernel(const float* __restrict__ input, float* output, int inner_size, int stride) {
    // Each block corresponds to one (outer, inner) pair
    int block_idx = blockIdx.x;
    int outer_idx = block_idx / inner_size;
    int inner_idx = block_idx % inner_size;

    // Compute the base offset for this (outer, inner) column
    int base_offset = outer_idx * stride * inner_size + inner_idx;

    // Allocate shared memory dynamically. The number of threads (n) is a power-of-two >= stride.
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Load element from global memory to shared memory if within the bounds of stride
    float val = 0.0f;
    if (tid < stride) {
        int g_idx = base_offset + tid * inner_size;
        val = input[g_idx];
        sdata[tid] = val;
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    int n = blockDim.x;  // n is a power-of-two >= stride

    // Blelloch scan (up-sweep phase) for exclusive scan
    for (int offset = 1; offset < n; offset *= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < n) {
            sdata[index] += sdata[index - offset];
        }
        __syncthreads();
    }

    // Set the last element to zero for the down-sweep phase
    if (tid == n - 1) {
        sdata[n - 1] = 0.0f;
    }
    __syncthreads();

    // Down-sweep phase
    for (int offset = n / 2; offset >= 1; offset /= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < n) {
            float temp = sdata[index - offset];
            sdata[index - offset] = sdata[index];
            sdata[index] += temp;
        }
        __syncthreads();
    }

    // Convert exclusive scan result to inclusive scan by adding the original value
    if (tid < stride) {
        float incl = sdata[tid] + val;
        int g_idx = base_offset + tid * inner_size;
        output[g_idx] = incl;
    }
}

// Host function to set up dimensions and launch the kernel
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    // Calculate outer_size: product of dimensions before 'dim'
    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }

    // Calculate inner_size: product of dimensions after 'dim'
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= x.size(i);
    }

    // Number of elements along the cumsum dimension
    int stride = x.size(dim);

    // Determine the number of threads per block as the next power-of-two >= stride
    int threads = 1;
    while (threads < stride) {
        threads *= 2;
    }

    // Each block processes one (outer, inner) pair
    int blocks = outer_size * inner_size;

    // Allocate shared memory (in bytes) for the scan
    size_t shared_memory_bytes = threads * sizeof(float);

    cumsum_shared_kernel<<<blocks, threads, shared_memory_bytes>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        inner_size,
        stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum using shared memory for parallel scan");
}
