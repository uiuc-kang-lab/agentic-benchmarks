#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Each warp computes the product reduction over the dimension of size 50.
// For each output element, one warp cooperatively loads the 50 elements:
// each thread in the warp processes indices starting at its lane id with a stride of 32.
// The warp-level reduction is then performed using __shfl_down_sync.

__global__ void prod_reduce_kernel(const float* __restrict__ input, float* __restrict__ output, int stride, int num_elements) {
    // Compute global warp id: each warp processes one output element
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // divide by 32
    int lane = threadIdx.x & 31;

    if (warp_id < num_elements) {
        float local_prod = 1.0f;
        // Each thread in the warp processes part of the 50 elements
        #pragma unroll
        for (int i = lane; i < 50; i += 32) {
            local_prod *= input[warp_id + i * stride];
        }
        // Perform warp-level reduction using __shfl_down_sync
        for (int offset = 16; offset > 0; offset /= 2) {
            local_prod *= __shfl_down_sync(0xffffffff, local_prod, offset);
        }

        // The first lane writes the result
        if (lane == 0) {
            output[warp_id] = local_prod;
        }
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    TORCH_CHECK(dim_size == 50, "Dimension size must be 50 for 50_Product_reduction");
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Choose a block size that is a multiple of the warp size
    int threads_per_block = 128;
    int warps_per_block = threads_per_block / 32;
    int blocks = (num_elements + warps_per_block - 1) / warps_per_block;

    prod_reduce_kernel<<<blocks, threads_per_block>>>(input_ptr, output_ptr, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA) with warp-level primitives");
}
