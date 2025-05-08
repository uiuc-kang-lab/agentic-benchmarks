#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel: Computes element-wise GELU activation and accumulates a global sum of outputs
// The atomicAdd is used only once per block after a per-block reduction in shared memory,
// thereby minimizing global atomic operations and contention.

__global__ void gelu_kernel_atomic_sum(const float* __restrict__ x, 
                                         float* __restrict__ y, 
                                         float* d_sum, 
                                         int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    
    float block_sum = 0.0f;
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    // Grid-stride loop for element-wise GELU computation
    for (int i = idx; i < n; i += stride) {
        float xi = x[i];
        float xi_cubed = xi * xi * xi;
        float inner = (xi + coeff * xi_cubed) * sqrt_2_over_pi;
        float result = 0.5f * xi * (1.0f + tanhf(inner));
        y[i] = result;
        block_sum += result;
    }

    // Each thread writes its partial block sum to shared memory
    sdata[tid] = block_sum;
    __syncthreads();

    // Reduction in shared memory to sum up the block's contributions
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Use atomic operation once per block to update the global sum
    if (tid == 0) {
        atomicAdd(d_sum, sdata[0]);
    }
}

// Host function to launch the kernel
// The function computes the GELU activation and also accumulates the global sum of outputs
// using minimal atomic operations. The returned tensor 'y' holds the correct element-wise results.

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();
    
    // Allocate a tensor for the global sum and initialize to 0
    auto sum_tensor = torch::zeros({1}, x.options());

    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(float);

    gelu_kernel_atomic_sum<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        n
    );

    // Optionally, one could use sum_tensor for further fused reduction operations or debugging.
    // Here, we return the element-wise GELU activation output as the correct result.
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA with minimal atomic operations");
}
