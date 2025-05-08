#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel: computes HardSigmoid activation using shared memory tiling
// Each block loads a tile of input values into shared memory, computes the activation, and writes results to global memory.
// HardSigmoid formula: y = clamp((x + 3) / 6, 0, 1)

template <typename scalar_t>
__global__ void hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    extern __shared__ char smem[];
    // Shared memory tile for the block
    scalar_t* tile = reinterpret_cast<scalar_t*>(smem);
    
    // Each thread in the block processes one element from the tile
    int tid = threadIdx.x;
    
    // Grid-stride loop to cover all elements
    for (size_t i = blockIdx.x * blockDim.x + tid; i < numel; i += blockDim.x * gridDim.x) {
        // Load one element from global memory into shared memory
        tile[tid] = input[i];
        __syncthreads();  // Ensure the entire tile is loaded before processing
        if (tid == 0) tile[0] = input[i];  // Load one element from global memory into shared memory
        
        // Compute the HardSigmoid activation from shared memory
        scalar_t x = tile[tid];
        scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
        y = (y < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) : ((y > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1) : y);
        
        // Write the result back to global memory
        output[i] = y;
        __syncthreads();  // Ensure all threads have finished processing the current tile before reuse
    }
}


// Host function to launch the CUDA kernel
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    
    size_t numel = input.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cuda", ([&] {
        size_t shmem_size = threads * sizeof(scalar_t);
        hardsigmoid_kernel<scalar_t><<<blocks, threads, shmem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA)");
}
