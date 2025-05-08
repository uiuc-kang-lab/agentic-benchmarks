#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using asynchronous copy (cp.async) with minimal synchronization
// This kernel is specialized for float type. Each thread loads 4 floats (16 bytes) from global memory
// into shared memory using cp.async, then computes tanhf, and finally writes the result back to global memory.

__global__ void tanh_async_kernel_float(const float* __restrict__ input, 
                                          float* __restrict__ output,
                                          const int numel) {
    // Allocate shared memory tile. Each block loads blockDim.x * 4 floats.
    extern __shared__ float tile[];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x * 4;
    int global_base = block_offset + tid * 4;

    // Determine if this thread can load a full vector of 4 floats
    bool full_load = (global_base + 3 < numel);

    if (global_base < numel) {
        if (full_load) {
            // Use cp.async to asynchronously load 16 bytes (4 floats) from global memory into shared memory
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                         :
                         : "r"(tile + tid * 4), "l"(input + global_base), "n"(16)
                         );
        } else {
            // For boundary cases, load available elements individually
            for (int j = 0; j < 4; j++) {
                if (global_base + j < numel) {
                    tile[tid * 4 + j] = input[global_base + j];
                }
            }
        }
    }

    // Wait for all asynchronous copies initiated by the block to complete
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    // Synchronize threads to ensure that shared memory is properly populated
    __syncthreads();

    // Process the loaded data
    if (global_base < numel) {
        if (full_load) {
            // Process 4 elements at once using vectorized loads/stores
            float4 data = reinterpret_cast<float4*>(tile)[tid];
            data.x = tanhf(data.x);
            data.y = tanhf(data.y);
            data.z = tanhf(data.z);
            data.w = tanhf(data.w);
            reinterpret_cast<float4*>(output)[(block_offset / 4) + tid] = data;
        } else {
            // For boundary cases, process elements individually
            for (int j = 0; j < 4; j++) {
                if (global_base + j < numel) {
                    output[global_base + j] = tanhf(tile[tid * 4 + j]);
                }
            }
        }
    }
}

// Generic kernel for non-float types; uses a simple grid-stride loop with standard tanh computation
template <typename scalar_t>
__global__ void tanh_generic_kernel(const scalar_t* __restrict__ input, 
                                      scalar_t* __restrict__ output, 
                                      const int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += stride) {
        output[i] = tanh(input[i]);
    }
}

// Forward function exposed to Python
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int numel = input.numel();
    const int threads = 256;

    if (input.scalar_type() == at::ScalarType::Float) {
        // Each block processes a tile of (threads * 4) elements
        int tile_elements = threads * 4;
        int blocks = (numel + tile_elements - 1) / tile_elements;
        size_t shared_mem = threads * 4 * sizeof(float);
        
        tanh_async_kernel_float<<<blocks, threads, shared_mem>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            numel
        );
    } else {
        int blocks = (numel + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_generic_kernel", ([&] {
            tanh_generic_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel
            );
        }));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Tanh forward with asynchronous copy and minimal __syncthreads() (CUDA)");
}
