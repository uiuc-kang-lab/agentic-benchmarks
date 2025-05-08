#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel that computes the GELU activation and also reduces the output values into a global sum
// using atomic operations only once per block after a shared memory reduction.

__global__ void gelu_kernel_atomic(const float* __restrict__ x, float* __restrict__ y, int n, float* block_sum_global) {
    extern __shared__ float sdata[];  // shared memory for block reduction
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    float gelu_val = 0.0f;
    if (index < n) {
        float val = x[index];
        float x_cube = val * val * val;
        float inner = (val + 0.044715f * x_cube) * 0.7978845608f;
        gelu_val = 0.5f * val * (1.0f + tanhf(inner));
        y[index] = gelu_val;
    } else {
        gelu_val = 0.0f;
    }

    // Each thread writes its GELU result into shared memory
    sdata[tid] = gelu_val;
    __syncthreads();

    // Perform block-wise reduction in shared memory to compute the sum of GELU outputs in the block
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Only one atomicAdd per block, minimizing global memory contention
    if (tid == 0) {
        atomicAdd(block_sum_global, sdata[0]);
    }
}

// Host function to launch the kernel. In addition to computing the element-wise GELU
// activation (stored in y), the kernel also computes a global sum of all outputs
// using minimized atomic operations. The global sum could be used for further diagnostics.

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(float);

    // Allocate a tensor for the global sum and initialize it to 0
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto global_sum = torch::zeros({1}, options);

    gelu_kernel_atomic<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n,
        global_sum.data_ptr<float>()
    );

    // The computed global sum is stored in 'global_sum'. It is computed using atomic operations
    // only once per block, thus minimizing contention. The main output, y, is correctly computed.
    
    // For this operation, we return the GELU output tensor y.
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation with atomic reduction for sum");
}
