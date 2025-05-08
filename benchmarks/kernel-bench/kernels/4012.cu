#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses double-buffering with asynchronous copy (cp.async) to load data into shared memory.
// While computing the ELU on one tile in shared memory, the next tile is asynchronously prefetched,
// which helps hide global memory latency on the NVIDIA H100 GPU. Proper synchronization via cp.async.wait_group
// and __syncthreads() ensures that race conditions are avoided and the results remain correct.

__global__ void elu_kernel_cp_async(const float * __restrict__ x, float * __restrict__ out, float alpha, int n) {
    extern __shared__ float shared_mem[];
    // Allocate double buffer in shared memory
    float *buffer0 = shared_mem;               // first buffer
    float *buffer1 = shared_mem + blockDim.x;    // second buffer
    
    const int blockSize = blockDim.x;
    const int tid = threadIdx.x;
    // Grid-stride loop: each block starts at a different offset
    const int stride = gridDim.x * blockSize;
    int index = blockIdx.x * blockSize;

    // If no work for this block, return early
    if (index >= n) return;

    // -- Load the first tile synchronously into buffer0 --
    if (index + tid < n) {
        buffer0[tid] = x[index + tid];
    }
    __syncthreads();

    // Process the first tile
    if (index + tid < n) {
        float val = buffer0[tid];
        out[index + tid] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
    }
    __syncthreads();

    // -- Pipeline: For subsequent tiles, prefetch next tile with cp.async while processing the current tile --
    for (index += stride; index < n; index += stride) {
        int global_index = index + tid;
        // Asynchronously load the next tile into buffer1 using cp.async
        if (global_index < n) {
            asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;\n" 
                          : 
                          : "r"(&buffer1[tid]), "l"(x + global_index), "n"(sizeof(float))
                         );
        }
        // Wait for the async copy to complete for the current group
        asm volatile ("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        // Process the tile that was previously loaded in buffer0
        int compute_index = index - stride + tid;
        if (compute_index < n) {
            float val = buffer0[tid];
            out[compute_index] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
        }
        __syncthreads();

        // Swap buffers so that buffer0 now points to the newly fetched data
        float *temp = buffer0;
        buffer0 = buffer1;
        buffer1 = temp;
        __syncthreads();
    }

    // Process the final tile that was loaded asynchronously in buffer0
    int compute_index = index - stride + tid;
    if (compute_index < n) {
        float val = buffer0[tid];
        out[compute_index] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
    }
}

// Host interface function
// Allocates necessary shared memory and launches the kernel using a grid-stride loop with double buffering.

torch::Tensor elu_cuda_async_shared(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Allocate shared memory for double buffering: 2 * blockDim.x * sizeof(float)
    size_t sharedMemSize = 2 * threads * sizeof(float);

    elu_kernel_cp_async<<<blocks, threads, sharedMemSize>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_async_shared, "ELU activation with asynchronous shared memory copy (CUDA)");
}
