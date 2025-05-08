#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define NUM_STREAMS 4

__global__ void streamed_prod_reduce_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          const int dim_size,
                                          const int stride,
                                          const int chunk_offset,
                                          const int chunk_size) {
    __shared__ float shared_data[BLOCK_SIZE];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x + chunk_offset;
    
    // Early exit if this thread is processing beyond the chunk
    if (gid >= chunk_offset + chunk_size) return;
    
    float thread_prod = 1.0f;
    
    // Process input elements assigned to this thread
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        thread_prod *= input[gid + i * stride];
    }
    
    // Store in shared memory
    shared_data[tid] = thread_prod;
    __syncthreads();
    
    // Reduce within block using warp-level primitives
    if (tid < WARP_SIZE) {
        float warp_prod = 1.0f;
        
        // Each thread in first warp reduces its portion
        for (int i = tid; i < BLOCK_SIZE; i += WARP_SIZE) {
            warp_prod *= shared_data[i];
        }
        
        // Warp-level reduction
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            warp_prod *= __shfl_down_sync(0xffffffff, warp_prod, offset);
        }
        
        if (tid == 0) {
            output[gid] = warp_prod;
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
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Calculate chunk size for each stream
    int chunk_size = (num_elements + NUM_STREAMS - 1) / NUM_STREAMS;
    int threads = BLOCK_SIZE;
    
    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int chunk_offset = i * chunk_size;
        int current_chunk_size = min(chunk_size, num_elements - chunk_offset);
        if (current_chunk_size <= 0) break;
        
        int blocks = (current_chunk_size + threads - 1) / threads;
        
        streamed_prod_reduce_kernel<<<blocks, threads, 0, streams[i]>>>(
            input_ptr, output_ptr, dim_size, stride, chunk_offset, current_chunk_size
        );
    }
    
    // Synchronize and cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed product reduction (CUDA)");
}