#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void hinge_loss_kernel_atomic(const float* predictions, const float* targets, float* output, int n) {
    __shared__ float cache[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;

    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        temp += fmaxf(0.0f, 1.0f - predictions[i] * targets[i]);
    }

    cache[tid] = temp;
    __syncthreads();

    // Perform reduction in shared memory with warp-level unrolling
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile float* smem = cache;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    // Use atomic operation to accumulate results from each block
    if (tid == 0) {
        atomicAdd(output, cache[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::zeros({1}, predictions.options());

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    hinge_loss_kernel_atomic<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Compute the mean hinge loss
    auto mean = output.div_(n);
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward with atomic optimization");
}