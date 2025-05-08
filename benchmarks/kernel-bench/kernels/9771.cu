#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)
#define TILE_SIZE 32

__device__ __forceinline__ float compute_conv_pixel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int b, int c, int oh, int ow,
    int in_h, int in_w, int channels,
    int kernel_h, int stride, int padding, int dilation) {
    
    float sum = 0.0f;
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
            int weight_idx = c * kernel_h + kh;
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    return sum;
}

__global__ void hybrid_depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int channels, int in_h, int in_w,
    int out_h, int out_w, int kernel_h,
    int stride, int padding, int dilation) {
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    const int tile_row = blockIdx.y * TILE_SIZE;
    const int tile_col = blockIdx.x * TILE_SIZE;
    
    const int bc_idx = blockIdx.z;
    const int b = bc_idx / channels;
    const int c = bc_idx % channels;
    
    if (b >= batch) return;
    
    const int row_stride = TILE_SIZE / WARPS_PER_BLOCK;
    const int start_row = tile_row + warp_id * row_stride;
    const float bias_val = bias[c];
    
    #pragma unroll
    for (int row_offset = 0; row_offset < row_stride; row_offset++) {
        const int oh = start_row + row_offset;
        if (oh >= out_h) continue;
        
        for (int ow = tile_col + lane_id; ow < min(tile_col + TILE_SIZE, out_w); ow += WARP_SIZE) {
            float sum = compute_conv_pixel(
                input, weight, b, c, oh, ow,
                in_h, in_w, channels, kernel_h,
                stride, padding, dilation);
            
            sum += bias_val;
            const int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
            output[output_idx] = sum;
        }
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    x = x.contiguous();
    weight = weight.contiguous();
    
    const int batch = x.size(0);
    const int channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_h = weight.size(2);
    
    if (groups != channels)
        throw std::invalid_argument("Depthwise convolution requires groups == channels");
    
    at::Tensor bias_val = bias.has_value() && bias.value().defined() 
        ? bias.value().contiguous() 
        : at::zeros({channels}, x.options());
    
    const int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - 1) / stride + 1;
    
    auto output = at::empty({batch, channels, out_h, out_w}, x.options());
    
    dim3 grid(
        (out_w + TILE_SIZE - 1) / TILE_SIZE,
        (out_h + TILE_SIZE - 1) / TILE_SIZE,
        batch * channels
    );
    dim3 block(BLOCK_SIZE);
    
    hybrid_depthwise_conv2d_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_val.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, in_h, in_w,
        out_h, out_w, kernel_h,
        stride, padding, dilation
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise convolution forward");
}