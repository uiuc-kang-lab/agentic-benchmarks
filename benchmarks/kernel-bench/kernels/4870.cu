#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for L1 normalization using manual loop unrolling
__global__ void l1_norm_forward_kernel_unroll(const float* __restrict__ x,
                                               float* __restrict__ out,
                                               int N,
                                               int D) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int step = blockDim.x;
    register float sum = 0.0f;
    
    // Calculate base index once to reduce register pressure
    const int row_offset = row * D;
    
    // Use register hint for frequently accessed values
    #pragma unroll 4
    for (int col = tid; col < D; col += step) {
        sum += fabsf(x[row_offset + col]);
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Perform reduction in shared memory using warp-level unrolling
    for (int stride = step / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        volatile float *vsmem = sdata;
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
    }
    __syncthreads();

    float total_sum = sdata[0];
    if (threadIdx.x == 0 && total_sum == 0.0f) {
        total_sum = 1e-12f;
        sdata[0] = total_sum;
    }
    __syncthreads();
    total_sum = sdata[0];

    // Normalize the row using manual loop unrolling
    col = threadIdx.x;
    for (; col + 3 * step < D; col += unroll_factor * step) {
        int base = row * D + col;
        float norm0 = x[base] / total_sum;
        float norm1 = x[base + step] / total_sum;
        float norm2 = x[base + 2 * step] / total_sum;
        float norm3 = x[base + 3 * step] / total_sum;
        out[base] = norm0;
        out[base + step] = norm1;
        out[base + 2 * step] = norm2;
        out[base + 3 * step] = norm3;
    }
    for (; col < D; col += step) {
        int idx = row * D + col;
        out[idx] = x[idx] / total_sum;
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

    l1_norm_forward_kernel_unroll<<<N, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA) with loop unrolling");
}
