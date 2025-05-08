#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_SIZE 256
#define WARP_SIZE 32

// Constant memory for frequently accessed data
__constant__ float const_data[TILE_SIZE];

__global__ void constant_mem_prod_reduce_kernel(const float* __restrict__ input,
                                                float* __restrict__ output,
                                                const int dim_size,
                                                const int stride) {
    __shared__ float shared_data[TILE_SIZE];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;
    
    // Initialize partial product
    float thread_prod = 1.0f;
    
    // Process input in tiles to maximize shared memory usage
    for (int tile_start = 0; tile_start < dim_size; tile_start += TILE_SIZE) {
        // Reset shared memory for new tile
        shared_data[tid] = 1.0f;
        __syncthreads();
        
        // Load tile into shared memory with coalesced access
        const int tile_end = min(tile_start + TILE_SIZE, dim_size);
        for (int i = tile_start + tid; i < tile_end; i += blockDim.x) {
            shared_data[tid] *= input[bid + i * stride];
        }
        __syncthreads();
        
        // Reduce within tile
        if (tid < WARP_SIZE) {
            float warp_prod = 1.0f;
            // Each thread in first warp reduces a portion of shared memory
            for (int i = tid; i < TILE_SIZE; i += WARP_SIZE) {
                warp_prod *= shared_data[i];
            }
            
            // Warp-level reduction using shuffle
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                warp_prod *= __shfl_down_sync(0xffffffff, warp_prod, offset);
            }
            
            // First thread in warp updates the thread product
            if (lane_id == 0) {
                thread_prod *= warp_prod;
            }
        }
    }
    
    // Final warp-level reduction
    if (tid < WARP_SIZE) {
        float final_prod = __shfl_sync(0xffffffff, thread_prod, 0);
        if (tid == 0) {
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
    
    // Copy frequently accessed data to constant memory
    cudaMemcpyToSymbol(const_data, input_ptr, TILE_SIZE * sizeof(float));

    // Launch configuration optimized for shared memory usage
    int threads = TILE_SIZE;  // Match shared memory tile size
    int blocks = num_elements;
    
    constant_mem_prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Constant memory optimized product reduction (CUDA)");
}