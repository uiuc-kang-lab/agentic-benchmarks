#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int WARP_SIZE = 32;

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int IC,
    int OC,
    int H,
    int W
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int total_warps = B * OC * H * W;
    
    if (warp_id >= total_warps) return;

    const int lane = tid % WARP_SIZE;
    const int w = warp_id % W;
    const int h = (warp_id / W) % H;
    const int oc = (warp_id / (W * H)) % OC;
    const int b = warp_id / (W * H * OC);

    const int elements_per_thread = (IC + WARP_SIZE - 1) / WARP_SIZE;
    float sum = 0.0f;

    for (int i = 0; i < elements_per_thread; ++i) {
        int ic = lane * elements_per_thread + i;
        if (ic < IC) {
            const int x_idx = ((b * IC + ic) * H + h) * W + w;
            const int w_idx = oc * IC + ic;
            sum += x[x_idx] * weight[w_idx];
        }
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        const int out_idx = ((b * OC + oc) * H + h) * W + w;
        output[out_idx] = bias ? sum + bias[oc] : sum;
    }
}

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias->dim() == 1, "Bias must be 1D");
    }

    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);

    TORCH_CHECK(weight.size(1) == IC, "Input/output channel mismatch");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "Kernel must be 1x1");
    if (bias) TORCH_CHECK(bias->size(0) == OC, "Bias/out channel mismatch");

    auto output = torch::empty({B, OC, H, W}, x.options());
    const int total_warps = B * OC * H * W;
    const int threads = 256;
    const int blocks = (total_warps * WARP_SIZE + threads - 1) / threads;

    forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, IC, OC, H, W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution optimized with warp reduce");
}
