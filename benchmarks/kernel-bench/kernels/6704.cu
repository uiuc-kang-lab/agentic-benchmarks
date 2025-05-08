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
#define TILE_SIZE 4

__global__ void hybrid_prod_reduce_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        const int dim_size,
                                        const int stride) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    __shared__ float warp_results[NUM_WARPS];
    
    // Phase 1: Tiled computation with vectorized loads where possible
    float local_prod = 1.0f;
    float4 vec_data;
    
    // Vector loads for coalesced memory access
    #pragma unroll
    for (int i = tid; i < (dim_size / TILE_SIZE) * TILE_SIZE; i += BLOCK_SIZE) {
        vec_data = reinterpret_cast<const float4*>(input + bid + i * stride)[0];
        local_prod *= vec_data.x * vec_data.y * vec_data.z * vec_data.w;
    }
    
    // Handle remaining elements
    #pragma unroll
    for (int i = tid + (dim_size / TILE_SIZE) * TILE_SIZE; i < dim_size; i += BLOCK_SIZE) {
        local_prod *= input[bid + i * stride];
    }
    
    // Phase 2: Warp-level reduction using shuffle instructions
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        local_prod *= __shfl_down_sync(0xffffffff, local_prod, offset);
    }
    
    if (lane_id == 0) {
        warp_results[warp_id] = local_prod;
    }
    
    __syncthreads();
    
    // Final reduction using first warp
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
    
    hybrid_prod_reduce_kernel<<<num_elements, BLOCK_SIZE>>>(
        input_ptr, output_ptr, dim_size, stride);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid optimized product reduction (CUDA)");
}