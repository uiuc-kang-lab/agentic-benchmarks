#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int b_oc = blockIdx.z;
    
    const int b = b_oc / OC;
    const int oc = b_oc % OC;
    
    if (b >= B || oc >= OC || h >= H || w >= W) return;

    float sum = 0.0f;
    
    for (int ic = 0; ic < IC; ++ic) {
        const int x_idx = ((b * IC + ic) * H + h) * W + w;
        const int w_idx = oc * IC + ic;
        sum += x[x_idx] * weight[w_idx];
    }

    if (bias) {
        sum += bias[oc];
    }
    
    const int out_idx = ((b * OC + oc) * H + h) * W + w;
    output[out_idx] = sum;
}

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    
    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);

    auto output = torch::empty({B, OC, H, W}, x.options());

    const dim3 block(32, 4);  // Optimized for spatial coalescing
    const dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y,
        B * OC
    );

    forward_kernel<<<grid, block>>>(
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
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA optim)");
}