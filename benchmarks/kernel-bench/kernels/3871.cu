#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses double-buffering with asynchronous copies (cp.async) to overlap
// global memory loads with computation. The input is divided into tiles of size blockDim.x,
// and while the current tile is being processed from shared memory, the next tile is asynchronously
// loaded into the alternate shared memory buffer. This pipelining helps hide global memory latency
// on the NVIDIA H100 GPU.

__global__ void softsign_async_kernel(const float* __restrict__ x, float* __restrict__ out, int num_elements) {
    // Allocate shared memory space for double buffering. The launch configuration must pass 2 * blockDim.x * sizeof(float).
    extern __shared__ float shared_buffer[];
    float* buffer0 = shared_buffer;
    float* buffer1 = shared_buffer + blockDim.x;

    int block_start = blockIdx.x * blockDim.x;  // Starting global index for this block
    int tid = threadIdx.x;

    // Determine how many tiles this block will process
    int num_tiles = (num_elements - block_start + blockDim.x - 1) / blockDim.x;

    // Prefetch first tile into buffer0 asynchronously
    int global_index = block_start + tid;
    if (global_index < num_elements) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;"
                      :
                      : "r"(&buffer0[tid]), "l"(x + global_index), "n"(sizeof(float)));
    }
    // Commit the async copy group and wait for it to complete
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // Process each tile
    for (int tile = 0; tile < num_tiles; tile++) {
        // If there is a next tile, start prefetching it into the alternate buffer
        if (tile < num_tiles - 1) {
            int next_global_index = block_start + (tile + 1) * blockDim.x + tid;
            if (next_global_index < num_elements) {
                // Choose the alternate buffer: if current tile is in buffer0 then load into buffer1, and vice versa
                float* next_buffer = (tile & 1) ? buffer0 : buffer1;
                asm volatile("cp.async.cg.shared.global [%0], [%1], %2;"
                             :
                             : "r"(&next_buffer[tid]), "l"(x + next_global_index), "n"(sizeof(float)));
            }
            asm volatile("cp.async.commit_group;");
        }

        // Wait for the asynchronous copy for the current tile to complete before using it
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Select the current buffer where the tile was loaded
        float* current_buffer = (tile & 1) ? buffer1 : buffer0;
        int current_global_index = block_start + tile * blockDim.x + tid;
        if (current_global_index < num_elements) {
            float val = current_buffer[tid];
            // Compute softsign: x / (1 + |x|)
            float result = val / (1.0f + fabsf(val));
            out[current_global_index] = result;
        }

        __syncthreads();
    }
}

// The forward function partitions the work among blocks and provides the shared memory required for double buffering.
// This version overlaps the global memory loads with computation using CUDA's cp.async intrinsics.

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    // Define block and grid sizes
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    // Each block uses double buffering: 2 * threads * sizeof(float) shared memory
    int shared_mem_size = 2 * threads * sizeof(float);
    
    softsign_async_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with overlapped computation and memory transfers (CUDA)");
}
