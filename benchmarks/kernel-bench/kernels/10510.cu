#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel performs a cumulative sum (inclusive scan) along the specified dimension.
// It reorganizes the memory access pattern: each block handles one "vector" (one instance of the cumulative sum to compute)
// and loads the vector elements into shared memory. Global loads/stores are done with a stride that aligns consecutive threads
// to (nearly) consecutive global addresses, which improves memory coalescing on the GPU.

__global__ void cumsum_coalesced_kernel(const float* __restrict__ input, 
                                          float* __restrict__ output, 
                                          int stride, 
                                          int inner_size) {
    // Each cumulative sum vector is identified by an index combining outer and inner dims.
    int vector_id = blockIdx.x;
    int outer_idx = vector_id / inner_size;
    int inner_idx = vector_id % inner_size;

    // Compute the base offset for this vector in the flattened tensor.
    // For a tensor with shape [outer_size, stride, inner_size], the element at position i along the cumsum dimension
    // is located at: base + i * inner_size
    int base = outer_idx * stride * inner_size + inner_idx;

    extern __shared__ float sdata[]; // Shared memory to hold a vector of 'stride' elements

    int tid = threadIdx.x;
    if (tid < stride) {
        // Load the element from global memory. Note that the distance between elements is inner_size.
        sdata[tid] = input[base + tid * inner_size];
    }
    __syncthreads();

    // Perform an inclusive scan (cumulative sum) in shared memory using the Hillis-Steele algorithm.
    for (int d = 1; d < stride; d *= 2) {
        float val = sdata[tid];
        __syncthreads();
        if (tid >= d) {
            val += sdata[tid - d];
        }
        __syncthreads();
        sdata[tid] = val;
        __syncthreads();
    }

    if (tid < stride) {
        // Write the scanned value back to global memory, preserving the original strided layout.
        output[base + tid * inner_size] = sdata[tid];
    }
}

// The forward function sets up the kernel launch parameters based on the input tensor shape and the target dimension.
// It computes outer_size, inner_size, and stride so that each cumulative sum vector (of length 'stride') is processed
// by one block with 'stride' threads. Global memory accesses for loading and storing vector elements are arranged
// to be as coalesced as possible.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= x.size(i);
    }

    int stride = x.size(dim);

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= x.size(i);
    }

    // Each cumulative sum vector is determined by outer and inner indices.
    int num_vectors = outer_size * inner_size;

    // Launch one block per vector. Each block uses 'stride' threads and 'stride * sizeof(float)' bytes of shared memory.
    dim3 blocks(num_vectors);
    dim3 threads(stride);
    
    cumsum_coalesced_kernel<<<blocks, threads, stride * sizeof(float)>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        stride,
        inner_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with coalesced memory accesses");
}
