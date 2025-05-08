#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

__global__ void depthwise_conv2d_warp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int stride,
    int padding,
    int dilation) 
{
    // Calculate warp and lane IDs
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    
    // Calculate global position
    const int total_warps = gridDim.x * WARPS_PER_BLOCK;
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    // Process multiple elements per warp
    for (int idx = global_warp_id; idx < batch * channels * out_h; idx += total_warps) {
        const int b = idx / (channels * out_h);
        const int c = (idx / out_h) % channels;
        const int oh = idx % out_h;
        
        if (b < batch) {
            // Each lane in the warp processes different output width positions
            for (int ow = lane_id; ow < out_w; ow += WARP_SIZE) {
                float sum = 0.0f;
                
                #pragma unroll
                for (int kh = 0; kh < kernel_h; ++kh) {
                    const int ih = oh * stride - padding + kh * dilation;
                    const int iw = ow * stride - padding;
                    
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        const int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                        const int weight_idx = c * kernel_h + kh;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
                
                // Add bias - no need for warp reduction since each lane computes independent outputs
                sum += bias[c];
                
                // Write output
                const int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                output[output_idx] = sum;
            }
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
    int groups) 
{
    x = x.contiguous();
    weight = weight.contiguous();
    
    const int batch = x.size(0);
    const int channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_h = weight.size(2);
    
    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }
    
    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }
    
    const int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - 1) / stride + 1;
    
    auto output = at::empty({batch, channels, out_h, out_w}, x.options());
    
    // Calculate grid size based on problem dimensions
    const int total_warps_needed = batch * channels * out_h;
    const int blocks = (total_warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    
    depthwise_conv2d_warp_kernel<<<blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_val.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        stride,
        padding,
        dilation
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}