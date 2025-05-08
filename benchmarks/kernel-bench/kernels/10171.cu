#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define BLOCK_SIZE 128
#define BLOCK_DIM_X 32
#define VECTOR_SIZE 4

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels * out_h * out_w) return;

    const int ow = idx % out_w;
    const int oh = (idx / out_w) % out_h;
    const int c = (idx / (out_w * out_h)) % channels;
    const int n = idx / (out_w * out_h * channels);

    scalar_t sum = 0;
    #pragma unroll
    for (int i = 0; i < k; ++i) {
        #pragma unroll
        for (int j = 0; j < k; ++j) {
            const int ih = oh * stride - padding + i * dilation;
            const int iw = ow * stride - padding + j * dilation;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                sum += input[((n * channels + c) * in_h + ih) * in_w + iw] * 
                       weight[(c * k + i) * k + j];
            }
        }
    }
    output[idx] = bias ? sum + bias[c] : sum;
}

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int height,
    int width) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_ch = blockIdx.z;
    
    if (x >= width || y >= height || out_ch >= out_channels) return;

    using vec_t = typename std::conditional<std::is_same<scalar_t, float>::value, float4, double2>::type;
    const int vec_channels = in_channels / VECTOR_SIZE;
    
    scalar_t sum = 0;
    for (int b = 0; b < batch; ++b) {
        for (int ich = 0; ich < vec_channels; ++ich) {
            const vec_t* ivec = reinterpret_cast<const vec_t*>(&input[((b * in_channels + ich*VECTOR_SIZE) * height + y) * width + x]);
            const vec_t* wvec = reinterpret_cast<const vec_t*>(&weight[out_ch * in_channels + ich*VECTOR_SIZE]);
            
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; ++v) {
                sum += reinterpret_cast<const scalar_t*>(ivec)[v] * 
                       reinterpret_cast<const scalar_t*>(wvec)[v];
            }
        }
        // Handle remaining channels
        for (int ich = vec_channels*VECTOR_SIZE; ich < in_channels; ++ich) {
            sum += input[((b * in_channels + ich) * height + y) * width + x] * 
                   weight[out_ch * in_channels + ich];
        }
    }
    
    const int out_idx = ((out_ch * batch + blockIdx.z) * height + y) * width + x;
    output[out_idx] = bias ? sum + bias[out_ch] : sum;
}

torch::Tensor forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride,
    int padding,
    int dilation) {

    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int k = depthwise_weight.size(2);
    int out_h = (in_h + 2*padding - dilation*(k-1) - 1)/stride + 1;
    int out_w = (in_w + 2*padding - dilation*(k-1) - 1)/stride + 1;

    auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());
    
    // Depthwise kernel
    int total_dw = batch * in_channels * out_h * out_w;
    dim3 block_dw(BLOCK_SIZE);
    dim3 grid_dw((total_dw + block_dw.x - 1) / block_dw.x);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<grid_dw, block_dw>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            depthwise_bias.defined() ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
            depthwise_output.data_ptr<scalar_t>(),
            batch, in_channels, in_h, in_w, out_h, out_w,
            k, stride, padding, dilation);
    }));

    // Pointwise kernel
    int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

    dim3 block_pw(BLOCK_DIM_X, 1);
    dim3 grid_pw(
        (out_w + block_pw.x - 1) / block_pw.x,
        (out_h + block_pw.y - 1) / block_pw.y,
        out_channels
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
        pointwise_conv2d_kernel<scalar_t><<<grid_pw, block_pw>>>(
            depthwise_output.data_ptr<scalar_t>(),
            pointwise_weight.data_ptr<scalar_t>(),
            pointwise_bias.defined() ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch, in_channels, out_channels, out_h, out_w);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Optimized depthwise separable conv");
}
