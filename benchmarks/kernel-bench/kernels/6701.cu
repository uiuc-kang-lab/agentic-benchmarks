#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

__device__ __forceinline__ float warpReduceProd(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void two_level_prod_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    const int dim_size,
                                    const int stride) {
    __shared__ float warp_products[WARPS_PER_BLOCK];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    // First level: compute partial products with coalesced memory access
    float thread_prod = 1.0f;
    
    #pragma unroll 4
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        thread_prod *= input[bid + i * stride];
    }
    
    // First warp-level reduction
    thread_prod = warpReduceProd(thread_prod);
    
    // Store warp results
    if (lane_id == 0) {
        warp_products[warp_id] = thread_prod;
    }
    __syncthreads();
    
    // Second level: final reduction using first warp
    if (warp_id == 0) {
        float final_prod = 1.0f;
        if (lane_id < WARPS_PER_BLOCK) {
            final_prod = warp_products[lane_id];
        }
        
        // Final warp-level reduction
        final_prod = warpReduceProd(final_prod);
        
        // Write final result
        if (lane_id == 0) {
            output[bid] = final_prod;
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
    
    // Launch configuration optimized for occupancy
    dim3 blocks(num_elements);
    dim3 threads(BLOCK_SIZE);
    
    two_level_prod_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Two-level optimized product reduction (CUDA)");
}