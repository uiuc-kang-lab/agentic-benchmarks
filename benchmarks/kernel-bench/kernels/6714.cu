#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// This kernel performs a product reduction over a specified dimension with minimal synchronization.
// Each block computes one output element. Threads within a block accumulate a partial product,
// then perform an intra-warp reduction using shuffle instructions (which require no explicit sync).
// Only one __syncthreads() is used after writing one partial result per warp to shared memory,
// ensuring shared memory consistency before the final reduction is done by the first warp.

__global__ void minimal_sync_prod_reduction_kernel(const float* __restrict__ input,
                                                    float* __restrict__ output,
                                                    int dim_size,
                                                    int stride) {
    // Each block processes one output element
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // Each thread computes its local product over the reduction dimension
    float local_prod = 1.0f;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        local_prod *= input[bid + i * stride];
    }

    // Intra-warp reduction using shfl_down_sync for minimal overhead
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_prod *= __shfl_down_sync(0xffffffff, local_prod, offset);
    }

    // Store one partial result per warp in shared memory
    __shared__ float warp_partial[BLOCK_SIZE / WARP_SIZE];
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0) {
        warp_partial[warp_id] = local_prod;
    }

    // Synchronize only once to ensure all warp results are visible
    __syncthreads();

    // Final reduction by the first warp; since the number of warps is small and they map to the first warp,
    // no additional __syncthreads() calls are necessary within this warp.
    if (tid < (blockDim.x / WARP_SIZE)) {
        float final_val = warp_partial[tid];
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            final_val *= __shfl_down_sync(0xffffffff, final_val, offset);
        }
        if (tid == 0) {
            output[bid] = final_val;
        }
    }
}

// Forward function handling tensor shape and launching the kernel
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

    // Launch one block per output element with BLOCK_SIZE threads per block
    minimal_sync_prod_reduction_kernel<<<num_elements, BLOCK_SIZE>>>(input_ptr, output_ptr, dim_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Minimal synchronization product reduction over a dimension (CUDA)");
}
