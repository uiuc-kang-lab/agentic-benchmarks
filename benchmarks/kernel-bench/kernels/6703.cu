#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define CHUNK_SIZE 4  // Number of elements per thread to process at once

__global__ void strided_prod_reduce_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         const int dim_size,
                                         const int stride,
                                         const int total_chunks) {
    __shared__ float smem[BLOCK_SIZE];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;
    
    // Initialize thread local product
    float thread_prod = 1.0f;
    
    // Each thread processes multiple chunks strided across the input
    #pragma unroll 2
    for (int chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
        const int base_idx = chunk_idx * grid_stride + gid;
        if (base_idx < dim_size) {
            // Process CHUNK_SIZE elements at once if available
            #pragma unroll
            for (int i = 0; i < CHUNK_SIZE && (base_idx + i * grid_stride) < dim_size; i++) {
                const int idx = base_idx + i * grid_stride;
                thread_prod *= input[bid + idx * stride];
            }
        }
    }
    
    // Store in shared memory
    smem[tid] = thread_prod;
    __syncthreads();
    
    // Reduce within block using sequential addressing
    #pragma unroll
    for (int offset = BLOCK_SIZE/2; offset >= WARP_SIZE; offset >>= 1) {
        if (tid < offset) {
            smem[tid] *= smem[tid + offset];
        }
        __syncthreads();
    }
    
    // Final warp reduction
    if (tid < WARP_SIZE) {
        float warp_prod = smem[tid];
        
        // Warp-level reduction using shuffle
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            warp_prod *= __shfl_down_sync(0xffffffff, warp_prod, offset);
        }
        
        // Write result
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
    
    // Calculate number of chunks needed to process all elements
    int total_chunks = (dim_size + (BLOCK_SIZE * CHUNK_SIZE) - 1) / (BLOCK_SIZE * CHUNK_SIZE);
    
    // Launch kernel with optimized configuration
    int threads = BLOCK_SIZE;
    int blocks = num_elements;
    
    strided_prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride, total_chunks);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided product reduction (CUDA)");
}