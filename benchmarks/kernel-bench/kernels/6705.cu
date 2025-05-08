#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_SIZE 256
#define WARP_SIZE 32
#define MAX_BLOCKS_PER_SM 32

__global__ void hybrid_prod_reduce_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        const int dim_size,
                                        const int stride,
                                        const int items_per_thread) {
    __shared__ float shared_data[TILE_SIZE];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;
    
    // Initialize thread-local product
    float thread_prod = 1.0f;
    
    // Each thread processes multiple elements before reduction
    #pragma unroll 4
    for (int i = tid; i < dim_size; i += blockDim.x * items_per_thread) {
        float local_prod = 1.0f;
        #pragma unroll
        for (int j = 0; j < items_per_thread && (i + j * blockDim.x) < dim_size; j++) {
            local_prod *= input[bid + (i + j * blockDim.x) * stride];
        }
        thread_prod *= local_prod;
    }
    
    // Store in shared memory
    shared_data[tid] = thread_prod;
    __syncthreads();
    
    // Two-phase reduction: first at block level, then at warp level
    if (tid < WARP_SIZE) {
        float warp_prod = 1.0f;
        // Each thread in first warp reduces its portion
        for (int i = tid; i < TILE_SIZE; i += WARP_SIZE) {
            warp_prod *= shared_data[i];
        }
        
        // Warp-level reduction using shuffle operations
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
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
    
    // Optimize thread and block configuration
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    
    int threads = TILE_SIZE;
    int blocks = num_elements;
    int items_per_thread = (dim_size + (threads * MAX_BLOCKS_PER_SM - 1)) / (threads * MAX_BLOCKS_PER_SM);
    
    hybrid_prod_reduce_kernel<<<blocks, threads>>>(
        input_ptr, output_ptr, dim_size, stride, items_per_thread);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid tiled product reduction (CUDA)");
}