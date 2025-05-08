#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32

// This kernel performs product reduction over a specified dimension using warp-level primitives.
// Each block computes one output element. Threads compute partial products over the reduction
// dimension and perform an intra-warp reduction using __shfl_down_sync. The warp leaders then
// store their results into a small shared array, and the first warp performs a final reduction
// using warp-level shuffles, eliminating the need for extensive shared memory operations.
__global__ void prod_reduce_kernel(const float* input, float* output, int dim_size, int stride) {
    // Each block is assigned to one output element
    int out_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int block_threads = blockDim.x;

    // Each thread computes a partial product over the reduction dimension using a strided loop
    float partial = 1.0f;
    for (int i = tid; i < dim_size; i += block_threads) {
        partial *= input[out_idx + i * stride];
    }

    // Intra-warp reduction using warp shuffle primitives
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, partial, offset);
        partial *= other;
    }

    // Each warp's lane 0 holds the reduced product for that warp
    __shared__ float warp_prod[32];  // enough for up to 32 warps per block (1024 threads)
    if (lane == 0) {
        warp_prod[warp_id] = partial;
    }
    __syncthreads();

    // Final reduction: let the first warp combine the results from all warps
    int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (tid < WARP_SIZE) {
        // Each thread in the first warp loads a warp result if available
        float val = (tid < numWarps) ? warp_prod[tid] : 1.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            val *= other;
        }
        if (tid == 0) {
            output[out_idx] = val;
        }
    }
}

// Forward function: removes the reduction dimension from the input shape and launches one block per output element
// Each block reduces over the specified dimension using warp-level shuffles, minimizing shared memory usage.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    // Get input shape and remove the reduction dimension to form the output shape
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Launch one block per output element. Using 256 threads per block to cover large reduction sizes.
    int threads = 256;
    int blocks = num_elements;

    prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension using warp-level intrinsics (CUDA)");
}
