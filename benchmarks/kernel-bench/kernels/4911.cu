#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float row_l1_sum(const float* __restrict__ x_row, int D, float* shared) {
    float sum = 0.0f;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        sum += fabsf(x_row[col]);
    }

    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void l1_norm_forward_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int N,
                                       int D) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const float* x_row = x + row * D;

    float total_sum = row_l1_sum(x_row, D, sdata);

    if (threadIdx.x == 0) {
        total_sum = fmaxf(total_sum, 1e-12f);
        sdata[0] = total_sum;
    }
    __syncthreads();
    total_sum = sdata[0];

    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        out[row * D + col] = x_row[col] / total_sum;
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);

    int threads = std::min<int>(512, D);
    int shared_mem_size = threads * sizeof(float);

    l1_norm_forward_kernel<<<N, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA)");
}
