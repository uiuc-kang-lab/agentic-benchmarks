#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device helper functions for memory access patterns
__device__ __forceinline__ float fetch_input(
    const float* __restrict__ input,
    int n, int c, int h, int w,
    int C, int H, int W) {
    return (h >= 0 && h < H && w >= 0 && w < W) ? 
           input[n * C * H * W + c * H * W + h * W + w] : 0.0f;
}

__device__ __forceinline__ float fetch_weight(
    const float* __restrict__ weight,
    int oc, int ic, int kh, int kw,
    int IC, int K) {
    return weight[oc * (IC * K * K) + ic * (K * K) + kh * K + kw];
}

// Main convolution kernel combining best practices from both implementations
__global__ void conv2d_hybrid_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int IC, int OC, int H, int W,
    int OH, int OW, int K,
    int stride, int padding, int dilation,
    int groups) {

    // Use 2D thread blocks for better spatial locality
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Compute output position
    const int n = bz;
    const int oc = by;
    const int oh = (blockDim.y * bx + ty);
    const int ow = tx;

    if (oh >= OH || ow >= OW) return;

    const int group_size = IC / groups;
    const int group_id = oc / (OC / groups);
    const int ic_start = group_id * group_size;
    const int ic_end = ic_start + group_size;

    // Compute convolution
    float sum = bias ? bias[oc] : 0.0f;
    
    #pragma unroll 4
    for (int ic = ic_start; ic < ic_end; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            const int ih = oh * stride - padding + kh * dilation;
            
            #pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                const int iw = ow * stride - padding + kw * dilation;
                sum += fetch_input(input, n, ic, ih, iw, IC, H, W) *
                       fetch_weight(weight, oc, ic - ic_start, kh, kw, group_size, K);
            }
        }
    }

    // Write output
    const int out_idx = n * OC * OH * OW + oc * OH * OW + oh * OW + ow;
    output[out_idx] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) CHECK_INPUT(bias.value());

    const int N = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);
    const int K = weight.size(2);
    
    const int OH = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    const int OW = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    auto output = torch::zeros({N, OC, OH, OW}, x.options());

    // Optimize thread block configuration
    dim3 threads(32, 8);  // 256 threads per block
    dim3 blocks((OH + threads.y - 1) / threads.y,
                OC,
                N);

    conv2d_hybrid_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, IC, OC, H, W, OH, OW, K,
        stride, padding, dilation, groups);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid optimized CUDA convolution forward");
}