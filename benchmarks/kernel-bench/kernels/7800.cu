#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<int KERNEL_H, int KERNEL_W>
__global__ void conv2d_unrolled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation,
    const int groups) {

    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z;

    if (w >= out_width || h >= out_height || oc >= out_channels) return;

    const int group_out_channels = out_channels / groups;
    const int group = oc / group_out_channels;
    const int in_channels_per_group = in_channels / groups;

    #pragma unroll
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int c = 0; c < in_channels_per_group; ++c) {
            const int input_channel = group * in_channels_per_group + c;
            const float* weight_ptr = &weight[((oc * in_channels_per_group + c) * KERNEL_H) * KERNEL_W];
            
            if (KERNEL_H == 3 && KERNEL_W == 3) {
                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    const int in_y = h * stride - padding + kh * dilation;
                    if (in_y >= 0 && in_y < in_height) {
                        #pragma unroll
                        for (int kw = 0; kw < 3; ++kw) {
                            const int in_x = w * stride - padding + kw * dilation;
                            if (in_x >= 0 && in_x < in_width) {
                                const int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                                sum += input[input_idx] * weight_ptr[kh * 3 + kw];
                            }
                        }
                    }
                }
            }
            else if (KERNEL_H == 5 && KERNEL_W == 5) {
                #pragma unroll
                for (int kh = 0; kh < 5; ++kh) {
                    const int in_y = h * stride - padding + kh * dilation;
                    if (in_y >= 0 && in_y < in_height) {
                        #pragma unroll
                        for (int kw = 0; kw < 5; ++kw) {
                            const int in_x = w * stride - padding + kw * dilation;
                            if (in_x >= 0 && in_x < in_width) {
                                const int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                                sum += input[input_idx] * weight_ptr[kh * 5 + kw];
                            }
                        }
                    }
                }
            }
            else {
                #pragma unroll
                for (int kh = 0; kh < KERNEL_H; ++kh) {
                    const int in_y = h * stride - padding + kh * dilation;
                    if (in_y >= 0 && in_y < in_height) {
                        #pragma unroll
                        for (int kw = 0; kw < KERNEL_W; ++kw) {
                            const int in_x = w * stride - padding + kw * dilation;
                            if (in_x >= 0 && in_x < in_width) {
                                const int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                                sum += input[input_idx] * weight_ptr[kh * KERNEL_W + kw];
                            }
                        }
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        const int output_idx = ((b * out_channels + oc) * out_height + h) * out_width + w;
        output[output_idx] = sum;
    }
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
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    const dim3 block_size(16, 16);
    const dim3 grid_size(
        (out_width + block_size.x - 1) / block_size.x,
        (out_height + block_size.y - 1) / block_size.y,
        out_channels
    );

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    if (kernel_height == 3 && kernel_width == 3) {
        conv2d_unrolled_kernel<3, 3><<<grid_size, block_size>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, in_height, in_width,
            out_channels, out_height, out_width,
            stride, padding, dilation, groups
        );
    }
    else if (kernel_height == 5 && kernel_width == 5) {
        conv2d_unrolled_kernel<5, 5><<<grid_size, block_size>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, in_height, in_width,
            out_channels, out_height, out_width,
            stride, padding, dilation, groups
        );
    }
    else {
        conv2d_unrolled_kernel<7, 7><<<grid_size, block_size>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, in_height, in_width,
            out_channels, out_height, out_width,
            stride, padding, dilation, groups
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with Unrolled Loops");
}