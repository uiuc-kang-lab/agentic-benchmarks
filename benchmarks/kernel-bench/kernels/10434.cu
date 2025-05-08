#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses shared memory and a parallel inclusive scan (Hillis-Steele) to compute the cumulative sum
// for each independent vector (combination of outer and inner dimensions). Each block processes one vector
// of length 'stride'. The block dimension is padded to the next power-of-two to enable an efficient scan.

__global__ void cumsum_scan_kernel(const float* input, float* output, int outer_size, int inner_size, int stride) {
    // Each block processes one cumulative sum vector of length 'stride'
    int vector_id = blockIdx.x; // vector index in [0, outer_size * inner_size)
    int outer_idx = vector_id / inner_size;
    int inner_idx = vector_id % inner_size;
    int tid = threadIdx.x;  // runs from 0 to padded_stride-1; padded_stride = blockDim.x

    extern __shared__ float s_data[];

    // Load data from global memory into shared memory for the vector
    if (tid < stride) {
        int g_index = outer_idx * (stride * inner_size) + tid * inner_size + inner_idx;
        s_data[tid] = input[g_index];
    } else {
        s_data[tid] = 0.0f; // pad with 0
    }
    __syncthreads();

    // Perform inclusive scan using Hillis-Steele algorithm in shared memory
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float val = 0.0f;
        if (tid >= offset) {
            val = s_data[tid - offset];
        }
        __syncthreads();
        s_data[tid] += val;
        __syncthreads();
    }

    // Write the result back to global memory
    if (tid < stride) {
        int g_index = outer_idx * (stride * inner_size) + tid * inner_size + inner_idx;
        output[g_index] = s_data[tid];
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
    
    // Compute padded_stride as the next power of two greater or equal to stride
    int padded_stride = 1;
    while (padded_stride < stride) {
        padded_stride *= 2;
    }

    // Each vector (combination of outer and inner indices) will be processed by one block 
    int num_vectors = outer_size * inner_size;

    // Launch the kernel with padded_stride threads per block and allocate shared memory accordingly
    cumsum_scan_kernel<<<num_vectors, padded_stride, padded_stride * sizeof(float)>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with shared memory scan");
}
