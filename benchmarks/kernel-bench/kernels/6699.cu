#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 256
#define TILE_SIZE 32
#define ELEMENTS_PER_THREAD 8

__global__ void tiled_prod_reduce_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       const int dim_size,
                                       const int stride) {
    __shared__ float shared_tile[BLOCK_SIZE + TILE_SIZE];  // Extra space to avoid bank conflicts
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid % TILE_SIZE;
    
    // Initialize thread-local product
    float thread_prod = 1.0f;
    
    // Process input in tiles
    #pragma unroll 1
    for (int base = 0; base < dim_size; base += BLOCK_SIZE * ELEMENTS_PER_THREAD) {
        // Load multiple elements per thread into shared memory
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int idx = base + tid + i * BLOCK_SIZE;
            if (idx < dim_size) {
                shared_tile[tid + i * TILE_SIZE] = input[bid + idx * stride];
            } else {
                shared_tile[tid + i * TILE_SIZE] = 1.0f;
            }
        }
        __syncthreads();
        
        // Process the tile
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE && (base + i) < dim_size; i++) {
            thread_prod *= shared_tile[i];
        }
        __syncthreads();
    }
    
    // Store partial product in shared memory with padding to avoid bank conflicts
    shared_tile[tid] = thread_prod;
    __syncthreads();
    
    // Reduce within the block using sequential addressing to minimize bank conflicts
    #pragma unroll
    for (int s = BLOCK_SIZE/2; s > TILE_SIZE/2; s >>= 1) {
        if (tid < s) {
            shared_tile[tid] *= shared_tile[tid + s];
        }
        __syncthreads();
    }
    
    // Final warp reduction using shuffle
    if (tid < TILE_SIZE) {
        float warp_prod = shared_tile[tid];
        #pragma unroll
        for (int offset = TILE_SIZE/2; offset > 0; offset >>= 1) {
            warp_prod *= __shfl_down_sync(0xffffffff, warp_prod, offset);
        }
        
        if (tid == 0) {
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
    
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(num_elements);
    
    tiled_prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled shared memory product reduction (CUDA)");
}