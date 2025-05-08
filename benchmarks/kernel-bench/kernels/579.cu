#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

// Define block size based on experimental tuning on the H100 GPU
#define BLOCK_SIZE 512
#define CHUNK_SIZE 4  // Each thread processes 4 float4 elements

// Kernel using vectorized loads (float4) and thread coarsening for better instruction-level parallelism
__global__ void multiplyVectorizedKernel(const float4* __restrict__ A,
                                           float4* __restrict__ C,
                                           float s,
                                           int64_t count) {
    // Use shared memory to cache scalar value
    __shared__ float s_scalar;
    if (threadIdx.x == 0) {
        s_scalar = s;
    }
    __syncthreads();
    
    // Each thread processes multiple elements (thread coarsening)
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int base_idx = blockIdx.x * (blockDim.x * CHUNK_SIZE) + tid;
    
    // Prefetch data into registers for better instruction-level parallelism
    float4 a[CHUNK_SIZE];
    
    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; i++) {
        int idx = base_idx + i * stride;
        if (idx < count) {
            a[i] = A[idx];
            // Multiply and store using the cached scalar value
            a[i].x *= s_scalar;
            a[i].y *= s_scalar;
            a[i].z *= s_scalar;
            a[i].w *= s_scalar;
            C[idx] = a[i];
        }
    }
}

// Kernel to deal with the remaining elements if the total number is not divisible by 4.
__global__ void multiplyRemainderKernel(const float* __restrict__ A,
                                         float* __restrict__ C,
                                         float s,
                                         int64_t start,
                                         int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = start + idx;
    if (i < size) {
        C[i] = A[i] * s;
    }
}

// Forward function that selects the vectorized kernel path if both A and C are 16-byte aligned.
// Using an experiment-based block size (BLOCK_SIZE = 512) to help maximize occupancy on the H100 GPU.

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = BLOCK_SIZE;  // Experimentally tuned block size

    // Check alignment for vectorized memory access (float4 requires 16-byte alignment)
    bool aligned = ((reinterpret_cast<uintptr_t>(A.data_ptr<float>()) % sizeof(float4)) == 0) &&
                   ((reinterpret_cast<uintptr_t>(C.data_ptr<float>()) % sizeof(float4)) == 0);

    if (aligned && size >= 4) {
        int64_t count = size / 4;     // Number of float4 elements
        int remainder = size % 4;       // Remaining elements
        int blocks = (count + threads - 1) / threads;

        multiplyVectorizedKernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(A.data_ptr<float>()),
            reinterpret_cast<float4*>(C.data_ptr<float>()),
            s,
            count);

        if (remainder > 0) {
            int start = count * 4;
            int remBlocks = (remainder + threads - 1) / threads;
            multiplyRemainderKernel<<<remBlocks, threads>>>(
                A.data_ptr<float>(),
                C.data_ptr<float>(),
                s,
                start,
                size);
        }
    } else {
        int blocks = (size + threads - 1) / threads;
        multiplyRemainderKernel<<<blocks, threads>>>(A.data_ptr<float>(), C.data_ptr<float>(), s, 0, size);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Block size optimized matrix-scalar multiplication kernel");
}
