#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that computes a cumulative sum along 'stride' dimension for each (outer, inner) index
// Each block corresponds to one outer index, and threads cover inner indices in a grid-stride loop.
// The kernel uses __ldg for efficient read-only memory access. 
// Block size is dynamically chosen in the host function based on inner_size to optimize performance.
__global__ void cumsum_dynamic_block_kernel(const float* __restrict__ input, float* output, int outer_size, int inner_size, int stride) {
    int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    // Each thread processes its own set of inner indices
    for (int inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
        float sum = 0.0f;
        int base_idx = outer_idx * stride * inner_size + inner_idx;
        
        // Unroll the loop for cumulative sum over the 'stride' dimension
        #pragma unroll 16
        for (int s = 0; s < stride; ++s) {
            int idx = base_idx + s * inner_size;
            sum += __ldg(&input[idx]);
            output[idx] = sum;
        }
    }
}

// Host function: computes dimensions, selects optimial block size, and launches the kernel
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    // Compute outer_size (product of dimensions before 'dim')
    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }

    // Compute inner_size (product of dimensions after 'dim')
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= x.size(i);
    }

    // 'stride' represents the size of the cumulative dimension
    int stride = x.size(dim);

    // Dynamically select the optimal block size based on inner_size
    int block_size;
    if (inner_size <= 32)
        block_size = 32;
    else if (inner_size <= 64)
        block_size = 64;
    else if (inner_size <= 128)
        block_size = 128;
    else if (inner_size <= 256)
        block_size = 256;
    else
        block_size = 512;

    // Launch one block per outer index, with the dynamically chosen block size
    cumsum_dynamic_block_kernel<<<outer_size, block_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with dynamic block size");
}
