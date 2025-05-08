#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for L1 normalization using manual loop unrolling
__global__ void l1_norm_forward_kernel_min_sync(const float* __restrict__ x,
                                                  float* __restrict__ out,
                                                  int N,
                                                  int D) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    float sum = 0.0f;
    const int step = blockDim.x;
    const int unroll_factor = 4;
    int col = threadIdx.x;

    // Accumulate absolute values with manual unrolling.
    // Process groups of 4 elements per iteration when possible.
    for (; col + 3 * step < D; col += unroll_factor * step) {
        int base = row * D + col;
        sum += fabsf(x[base]) +
               fabsf(x[base + step]) +
               fabsf(x[base + 2 * step]) +
               fabsf(x[base + 3 * step]);
    }
    // Process any remaining elements if D is not a multiple of 4*step
    for (; col < D; col += step) {
        sum += fabsf(x[row * D + col]);
    }

    sdata[threadIdx.x] = sum;
    __syncthreads(); // Necessary before reduction

    // Perform reduction in shared memory using warp-level unrolling
    for (int stride = step / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
    }
    if (threadIdx.x < 32) {  // Final warp-level reduction
        volatile float *vsmem = sdata;
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
    }
    if (threadIdx.x == 0) { // Only one thread writes the result
        if (sdata[0] == 0.0f) {
            sdata[0] = 1e-12f;
        }
    }
    __syncthreads(); // Necessary to ensure total_sum is ready
    float total_sum = sdata[0];

    // Cache reciprocal of total_sum to replace division with multiplication
    float recip_sum = 1.0f / total_sum;
    
    // Normalize the row using manual loop unrolling
    col = threadIdx.x;
    for (; col + 3 * step < D; col += unroll_factor * step) {
        int base = row * D + col;
        float norm0 = x[base] * recip_sum;
        float norm1 = x[base + step] * recip_sum;
        float norm2 = x[base + 2 * step] * recip_sum;
        float norm3 = x[base + 3 * step] * recip_sum;
        out[base] = norm0;
        out[base + step] = norm1;
        out[base + 2 * step] = norm2;
        out[base + 3 * step] = norm3;
    }
    for (; col < D; col += step) {
        int idx = row * D + col;
        out[idx] = x[idx] * recip_sum;
    }
}

// The forward function exposed to PyTorch
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);
    int threads = std::min<int>(1024, D);
    int shared_mem_size = threads * sizeof(float);

    l1_norm_forward_kernel_min_sync<<<N, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA) with minimal synchronization");
}
