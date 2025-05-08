#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

__global__ void minimal_sync_prod_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       const int dim_size,
                                       const int stride) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    // Shared memory only for final warp results
    __shared__ float warp_results[NUM_WARPS];
    
    // Phase 1: Each thread computes local product with strided access
    float local_prod = 1.0f;
    #pragma unroll 4
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        local_prod *= input[bid + i * stride];
    }

    // Phase 2: Warp-level reduction using shuffle instructions
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        local_prod *= __shfl_down_sync(0xffffffff, local_prod, offset);
    }

    // Only first thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = local_prod;
    }

    // Single sync point needed before final reduction
    __syncthreads();

    // Final reduction: first warp combines all warp results
    if (warp_id == 0 && lane_id < NUM_WARPS) {
        float warp_prod = warp_results[lane_id];
        
        #pragma unroll
        for (int offset = NUM_WARPS/2; offset > 0; offset >>= 1) {
            warp_prod *= __shfl_down_sync(0xffffffff, warp_prod, offset);
        }

        if (lane_id == 0) {
            output[bid] = warp_prod;
        }
    }
}

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
    
    minimal_sync_prod_kernel<<<num_elements, BLOCK_SIZE>>>(
        input_ptr, output_ptr, dim_size, stride);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Minimal sync product reduction (CUDA)");
}