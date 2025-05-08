#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Tail kernel to handle remaining elements if n is not divisible by 4
__global__ void gelu_kernel_tail(const float* __restrict__ x, float* __restrict__ y, int start, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = start + i;
    if (idx < n) {
        float v = x[idx];
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        float x_cubed = v * v * v;
        float inner = (v + coeff * x_cubed) * sqrt_2_over_pi;
        y[idx] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

// Pipeline kernel using asynchronous copy (cp.async) to overlap global memory -> shared memory transfers with computation
// This kernel processes data in units of float4 (16 bytes) for vectorized loads
// Assumes n (number of elements) is large; tail elements (n % 4) are handled by gelu_kernel_tail

__global__ void gelu_kernel_pipe(const float* __restrict__ x, float* __restrict__ y, int num_vec, int n) {
    // Use external shared memory. We will reinterpret it as an array of float4.
    extern __shared__ char smem[];
    typedef float4 vec4;
    vec4* shared_tile = reinterpret_cast<vec4*>(smem);

    int tid = threadIdx.x;
    // Each block processes a contiguous chunk of vectorized elements
    // block_start is the index (in terms of float4 elements) of the first element of this block
    int block_start = blockIdx.x * blockDim.x;
    // Determine how many tiles (iterations) this block will process
    int tiles = ((num_vec - block_start) + blockDim.x - 1) / blockDim.x; 

    // Prefetch the first tile asynchronously into shared memory
    int index = block_start + tid;
    if (index < num_vec) {
        // cp.async copies 16 bytes (size of a float4) from global memory to shared memory
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], %2;\n"
            :
            : "r"(&shared_tile[tid]), "l"(&((const vec4*)x)[index]), "n"(16)
        );
    }
    asm volatile("cp.async.commit_group;" ::: "memory");

    // Pipeline loop: For each tile, prefetch the next tile (if any) while computing the current tile
    for (int t = 0; t < tiles; t++) {
        int next_index = block_start + (t + 1) * blockDim.x + tid;
        if (t < tiles - 1 && next_index < num_vec) {
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], %2;\n"
                :
                : "r"(&shared_tile[tid]), "l"(&((const vec4*)x)[next_index]), "n"(16)
            );
            asm volatile("cp.async.commit_group;" ::: "memory");
        }

        // Wait for all async copies in the group to complete
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();

        // Load the data from shared memory
        vec4 data = shared_tile[tid];

        // Compute GELU activation on each component of the vector
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        auto gelu_val = [=](float v) -> float {
            float x_cubed = v * v * v;
            float inner = (v + coeff * x_cubed) * sqrt_2_over_pi;
            return 0.5f * v * (1.0f + tanhf(inner));
        };

        vec4 result;
        result.x = gelu_val(data.x);
        result.y = gelu_val(data.y);
        result.z = gelu_val(data.z);
        result.w = gelu_val(data.w);

        // Write the computed result back to global memory
        int out_index = block_start + t * blockDim.x + tid;
        if (out_index < num_vec) {
            ((vec4*)y)[out_index] = result;
        }
        __syncthreads();
    }
}

// Host function that partitions the work and uses a CUDA stream to overlap kernel execution and memory operations
// The input tensor 'x' is assumed to be on CUDA and contiguous. The output tensor 'y' will contain the GELU-activated values.

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    // Process most of the data using vectorized (float4) pipelined kernel
    int num_vec = n / 4;  // number of float4 elements
    int remainder = n % 4;  // leftover elements

    const int threads = 256;
    int blocks = (num_vec + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(float4);  // each thread loads one float4 per tile

    // Create a CUDA stream to overlap kernel execution with asynchronous memory operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Launch the pipelined kernel on the provided stream
    gelu_kernel_pipe<<<blocks, threads, shared_mem, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_vec, n);

    // If there are leftover elements, process them with a simple kernel launch (on the same stream)
    if (remainder > 0) {
        int tail_start = num_vec * 4;
        int tail_blocks = (remainder + threads - 1) / threads;
        gelu_kernel_tail<<<tail_blocks, threads, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), tail_start, n);
    }

    // Synchronize the stream to ensure all operations are complete before returning the result
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation with pipelined memory transfers using CUDA streams");
}
