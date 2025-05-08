/*
 * This CUDA kernel performs an efficient product reduction over a specified dimension.
 * It combines manual loop unrolling for computing the product, unrolled shared memory reduction,
 * and warp-level reduction using __shfl_down_sync. The __restrict__ qualifiers help the compiler
 * optimize memory accesses. This version fuses ideas from two separate kernel implementations
 * to minimize synchronization overhead and loop overhead.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32

// Combined efficient product reduction kernel:
__global__ void prod_reduction_fast(const float* __restrict__ input, float* __restrict__ output, int dim_size, int stride) {
    extern __shared__ float sdata[];
    int out_idx = blockIdx.x;  // each block computes one output element
    int tid = threadIdx.x;

    // Each thread computes partial product over the reduction dimension using loop unrolling
    float partial = 1.0f;
    #pragma unroll
    for (int i = tid; i < dim_size; i += blockDim.x) {
        partial *= input[out_idx + i * stride];
    }
    sdata[tid] = partial;
    __syncthreads();

    // Unrolled reduction in shared memory; combining ideas from both kernels
    if (blockDim.x >= 512) {
        if (tid < 256) {
            sdata[tid] *= sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) {
            sdata[tid] *= sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) {
            sdata[tid] *= sdata[tid + 64];
        }
        __syncthreads();
    }

    // Warp-level reduction using shuffle operations (no __syncthreads needed within a warp)
    if (tid < WARP_SIZE) {
        float val = sdata[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val *= __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            output[out_idx] = val;
        }
    }
}

// Forward function for Torch extension, performing product reduction over the specified dimension
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    // Determine output shape and reduction dimension size
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    int threads = 256;
    int blocks = num_elements;  // one block per output element
    size_t shared_mem_size = threads * sizeof(float);

    prod_reduction_fast<<<blocks, threads, shared_mem_size>>>(input_ptr, output_ptr, dim_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fast product reduction over a dimension (CUDA)");
}
