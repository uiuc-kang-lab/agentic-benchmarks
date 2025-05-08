#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Utility macros for tensor checks
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define block and warp constants
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Optimized kernel combining manual loop unrolling and minimal sync for product reduction
__global__ void optimized_prod_kernel(const float * __restrict__ input,
                                        float * __restrict__ output,
                                        int dim_size,
                                        int stride) {
    // Calculate thread indices
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;  // Each block processes one output element
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    // Shared memory for storing warp-level partial results
    __shared__ float warp_results[NUM_WARPS];

    // Phase 1: Each thread computes its local product using strided access.
    float local_prod = 1.0f;
    #pragma unroll 4
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        local_prod *= input[bid + i * stride];
    }

    // Phase 2: Intra-warp reduction using fast shuffle instructions
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_prod *= __shfl_down_sync(0xffffffff, local_prod, offset);
    }

    // Each warp's first thread writes its result to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = local_prod;
    }
    __syncthreads();

    // Phase 3: Final reduction by the first warp
    if (warp_id == 0 && lane_id < NUM_WARPS) {
        float warp_prod = warp_results[lane_id];
        for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
            warp_prod *= __shfl_down_sync(0xffffffff, warp_prod, offset);
        }
        if (lane_id == 0) {
            output[bid] = warp_prod;
        }
    }
}

// Forward function that sets up the reduction over the specified dimension
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    // Remove the reduction dimension for the output
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float *input_ptr = x.data_ptr<float>();
    float *output_ptr = output.data_ptr<float>();

    // Launch one block per output element
    optimized_prod_kernel<<<num_elements, BLOCK_SIZE>>>(input_ptr, output_ptr, dim_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized product reduction over a dimension (CUDA)");
}
