#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

__device__ __forceinline__ float warp_reduce_product(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void divergence_free_prod_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          const int dim_size,
                                          const int stride) {
    __shared__ float warp_products[WARPS_PER_BLOCK];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int warp_id = tid >> 5;
    
    // Each thread starts with multiplicative identity
    float thread_prod = 1.0f;
    
    // Uniform stride pattern for coalesced memory access
    // Using grid-stride loop to handle arbitrary input sizes
    const int grid_stride = blockDim.x * gridDim.x;
    #pragma unroll 4
    for (int idx = tid; idx < dim_size; idx += grid_stride) {
        const float val = input[bid + idx * stride];
        thread_prod *= val;
    }
    
    // Warp-level reduction without divergent branches
    thread_prod = warp_reduce_product(thread_prod);
    
    // Store warp results using predicated write
    const bool is_warp_leader = (lane_id == 0);
    if (is_warp_leader) {
        warp_products[warp_id] = thread_prod;
    }
    
    __syncthreads();
    
    // Final reduction by first warp only
    if (warp_id == 0) {
        // Load warp product or identity value based on lane ID
        const float warp_val = (lane_id < WARPS_PER_BLOCK) ? warp_products[lane_id] : 1.0f;
        
        // Final warp-level reduction
        const float final_prod = warp_reduce_product(warp_val);
        
        // Single thread writes final result
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
    
    // Launch configuration
    const int blocks = num_elements;
    const int threads = BLOCK_SIZE;
    
    divergence_free_prod_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Divergence-free product reduction (CUDA)");
}