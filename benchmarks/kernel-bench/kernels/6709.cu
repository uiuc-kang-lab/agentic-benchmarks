#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// This kernel performs product reduction over a dimension by accumulating partial products
// in registers, storing them in shared memory, then performing a tree-based reduction
// followed by a warp-level reduction using __shfl_down_sync().
__global__ void warp_shared_prod_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int dim_size,
                                          int stride) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int bid = blockIdx.x; // Each block processes one output element

    // Each thread computes its partial product over the reduction dimension
    float local_prod = 1.0f;
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        local_prod *= input[bid + i * stride];
    }

    // Store the partial product into shared memory
    sdata[tid] = local_prod;
    __syncthreads();

    // Intra-block reduction using shared memory
    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    // Final reduction using warp-level primitives
    if (tid < 32) {
        float val = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val *= __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            output[bid] = val;
        }
    }
}

// Forward function to set up kernel launch parameters
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    // Remove the reduced dimension from the output shape
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Launch one block per output element
    warp_shared_prod_kernel<<<num_elements, BLOCK_SIZE>>>(input_ptr, output_ptr, dim_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp and shared memory optimized product reduction (CUDA)");
}
