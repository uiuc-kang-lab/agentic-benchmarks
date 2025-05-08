#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<int KERNEL_SIZE>
__global__ void conv2d_unrolled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation) {

    const int n = blockIdx.x;
    const int oc = blockIdx.y;
    const int out_y = blockIdx.z * blockDim.y + threadIdx.y;
    const int out_x = threadIdx.x;

    if (out_y >= out_height || out_x >= out_width) return;

    float sum = 0.0f;
    const int in_y_base = out_y * stride - padding;
    const int in_x_base = out_x * stride - padding;

    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        // Manual unroll for kernel size 3x3
        if (KERNEL_SIZE == 3) {
            const float* in_ptr = input + (n * in_channels + ic) * in_height * in_width;
            const float* w_ptr = weight + (oc * in_channels + ic) * KERNEL_SIZE * KERNEL_SIZE;

            #define PROCESS_KERNEL(ky, kx) \
                { \
                    const int in_y = in_y_base + (ky) * dilation; \
                    const int in_x = in_x_base + (kx) * dilation; \
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) { \
                        sum += in_ptr[in_y * in_width + in_x] * w_ptr[(ky) * KERNEL_SIZE + (kx)]; \
                    } \
                }

            PROCESS_KERNEL(0, 0);
            PROCESS_KERNEL(0, 1);
            PROCESS_KERNEL(0, 2);
            PROCESS_KERNEL(1, 0);
            PROCESS_KERNEL(1, 1);
            PROCESS_KERNEL(1, 2);
            PROCESS_KERNEL(2, 0);
            PROCESS_KERNEL(2, 1);
            PROCESS_KERNEL(2, 2);

            #undef PROCESS_KERNEL
        }
        // Manual unroll for kernel size 5x5
        else if (KERNEL_SIZE == 5) {
            const float* in_ptr = input + (n * in_channels + ic) * in_height * in_width;
            const float* w_ptr = weight + (oc * in_channels + ic) * KERNEL_SIZE * KERNEL_SIZE;

            #define PROCESS_KERNEL(ky, kx) \
                { \
                    const int in_y = in_y_base + (ky) * dilation; \
                    const int in_x = in_x_base + (kx) * dilation; \
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) { \
                        sum += in_ptr[in_y * in_width + in_x] * w_ptr[(ky) * KERNEL_SIZE + (kx)]; \
                    } \
                }

            #pragma unroll
            for (int ky = 0; ky < 5; ky++) {
                #pragma unroll
                for (int kx = 0; kx < 5; kx++) {
                    PROCESS_KERNEL(ky, kx);
                }
            }

            #undef PROCESS_KERNEL
        }
        // Generic case
        else {
            const float* in_ptr = input + (n * in_channels + ic) * in_height * in_width;
            const float* w_ptr = weight + (oc * in_channels + ic) * KERNEL_SIZE * KERNEL_SIZE;

            #pragma unroll
            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                #pragma unroll
                for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                    const int in_y = in_y_base + ky * dilation;
                    const int in_x = in_x_base + kx * dilation;
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        sum += in_ptr[in_y * in_width + in_x] * w_ptr[ky * KERNEL_SIZE + kx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[(n * out_channels + oc) * out_height * out_width + out_y * out_width + out_x] = sum;
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

    TORCH_CHECK(groups == 1, "Only groups==1 is supported");

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    dim3 threads(32, 8);
    dim3 blocks(batch, out_channels, (out_height + threads.y - 1) / threads.y);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    if (kernel_size == 3) {
        conv2d_unrolled_kernel<3><<<blocks, threads>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch, in_channels, out_channels,
            in_height, in_width, out_height, out_width,
            stride, padding, dilation);
    }
    else if (kernel_size == 5) {
        conv2d_unrolled_kernel<5><<<blocks, threads>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch, in_channels, out_channels,
            in_height, in_width, out_height, out_width,
            stride, padding, dilation);
    }
    else {
        conv2d_unrolled_kernel<7><<<blocks, threads>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch, in_channels, out_channels,
            in_height, in_width, out_height, out_width,
            stride, padding, dilation);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fully unrolled CUDA convolution implementation");
}