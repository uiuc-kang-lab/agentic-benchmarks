#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel performs product reduction over a given dimension with manual loop unrolling
// to reduce loop overhead in both the product computation and the reduction phases.

__global__ void prod_reduce_kernel(const float* input, float* output, int dim_size, int stride) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int out_idx = blockIdx.x;  // Each block is responsible for one output element

    float product = 1.0f;
    // Compute partial product for this output element; unroll the loop to reduce overhead
    #pragma unroll
    for (int i = tid; i < dim_size; i += blockDim.x) {
        product *= input[out_idx + i * stride];
    }
    sdata[tid] = product;
    __syncthreads();

    // Manually unrolled reduction in shared memory
    if (blockDim.x >= 512 && tid < 256) { sdata[tid] *= sdata[tid + 256]; } __syncthreads();
    if (blockDim.x >= 256 && tid < 128) { sdata[tid] *= sdata[tid + 128]; } __syncthreads();
    if (blockDim.x >= 128 && tid < 64)  { sdata[tid] *= sdata[tid + 64]; } __syncthreads();

    // Warp-level reduction using unrolled loop; no __syncthreads needed within a warp
    if (tid < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sdata[tid] *= __shfl_down_sync(0xffffffff, sdata[tid], offset);
        }
    }

    if (tid == 0) {
        output[out_idx] = sdata[0];
    }
}

// Forward function that sets up the reduction over the specified dimension
// It removes the reduction dimension from the output shape and launches one block per output element.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    int threads = 256;
    int blocks = num_elements;
    size_t shared_mem_size = threads * sizeof(float);

    prod_reduce_kernel<<<blocks, threads, shared_mem_size>>>(input_ptr, output_ptr, dim_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension with manual loop unrolling (CUDA)");
}
