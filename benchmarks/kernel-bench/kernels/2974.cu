#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Device-specific tanh: use tanhf for float and tanh for double
template <typename scalar_t>
__device__ inline scalar_t device_tanh(scalar_t x);

template <>
__device__ inline float device_tanh<float>(float x) {
    return tanhf(x);
}

template <>
__device__ inline double device_tanh<double>(double x) {
    return tanh(x);
}

// Async double-buffered kernel using cp.async for asynchronous copy from global to shared memory. 
// This kernel overlaps global memory loads with tanh computation and uses __syncthreads() only where needed for shared memory consistency.
// It processes the input in tiles of blockDim.x elements per block, with double buffering to hide latency.

template <typename scalar_t>
__global__ void tanh_kernel_async(const scalar_t* __restrict__ input,
                                    scalar_t* __restrict__ output,
                                    int size) {
    // Allocate two buffers in shared memory for double buffering
    extern __shared__ char shared_mem[];  // total shared memory size: 2 * blockDim.x * sizeof(scalar_t)
    int tile_elems = blockDim.x;
    scalar_t* tile0 = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* tile1 = reinterpret_cast<scalar_t*>(shared_mem + tile_elems * sizeof(scalar_t));

    // Setup pointers for double buffering
    scalar_t* tile[2];
    tile[0] = tile0;
    tile[1] = tile1;

    int tid = threadIdx.x;
    // Global stride: each block processes tiles spaced by gridDim.x * blockDim.x
    int global_tile_stride = gridDim.x * blockDim.x;
    // Base index for this block
    int base_index = blockIdx.x * blockDim.x;
    
    // Initialize double buffering indices
    int curr = 0;
    int next = 1;

    // Preload first tile asynchronously from global memory into shared memory using cp.async
    int global_index = base_index;
    if (global_index + tid < size) {
        asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;\n"
                      :
                      : "r"(tile[curr] + tid), "l"(input + global_index + tid), "n"(sizeof(scalar_t))
                      : "memory");
    }
    // Commit the async copy group
    asm volatile("cp.async.commit_group;\n");
    // Synchronize only once to ensure the first tile is available
    __syncthreads();

    // Process tiles in a grid-stride loop with double buffering
    for (int pos = base_index; pos < size; pos += global_tile_stride) {
        // Compute tanh on the current tile stored in shared memory
        if (pos + tid < size) {
            tile[curr][tid] = device_tanh(tile[curr][tid]);
        }
        
        // Initiate async copy for the next tile if it exists
        int next_pos = pos + global_tile_stride;
        if (next_pos + tid < size) {
            asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;\n"
                          :
                          : "r"(tile[next] + tid), "l"(input + next_pos + tid), "n"(sizeof(scalar_t))
                          : "memory");
        }
        asm volatile("cp.async.commit_group;\n");
        
        // Write out the computed results from the current tile to global memory
        if (pos + tid < size) {
            output[pos + tid] = tile[curr][tid];
        }
        
        // Synchronize to ensure that the next tile has finished loading before processing it in the next iteration
        __syncthreads();
        
        // Swap buffers for double buffering
        int temp = curr;
        curr = next;
        next = temp;
    }
}

// Host function to launch the asynchronous double-buffered tanh kernel
// Note: Shared memory size allocated is 2 * blockDim.x * sizeof(scalar_t)

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads = 256;
    // Each block processes a tile of 'threads' elements; choose enough blocks to cover the input
    int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_async", ([&] {
        tanh_kernel_async<scalar_t><<<blocks, threads, 2 * threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Async double-buffered Tanh forward (CUDA)");
}
