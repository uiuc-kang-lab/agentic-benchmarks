#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32

// This kernel performs a product reduction over a specified dimension using only warp-level primitives.
// Each block is launched with exactly one warp (32 threads) to compute one output element.
// Each thread computes a partial product over the reduction dimension and then the partial products are
// combined using __shfl_down_sync, eliminating any shared memory usage.
__global__ void warp_only_prod_reduction_kernel(const float* __restrict__ input,
                                                 float* __restrict__ output,
                                                 int dim_size,
                                                 int stride) {
    // Each block computes one output element
    int outIdx = blockIdx.x;
    int lane = threadIdx.x;  // thread index within the warp (0 to 31)

    float partial = 1.0f;
    // Compute partial product over the reduction dimension in a strided loop
    for (int i = lane; i < dim_size; i += WARP_SIZE) {
        partial *= input[outIdx + i * stride];
    }

    // Perform warp-level reduction using __shfl_down_sync in a branch-free manner
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        partial *= __shfl_down_sync(0xffffffff, partial, offset);
    }

    // The final reduced product is in lane 0
    if (lane == 0) {
        output[outIdx] = partial;
    }
}

// The forward function removes the reduction dimension from the input shape and launches one block per output element
// using exactly one warp (32 threads) per block. This eliminates shared memory usage and leverages fast warp-level shuffles.
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

    int threads = WARP_SIZE; // 32 threads per block, one warp
    int blocks = num_elements; // one block per output element

    warp_only_prod_reduction_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA) using only warp-level primitives");
}
