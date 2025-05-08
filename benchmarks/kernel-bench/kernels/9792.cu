#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define MAX_KERNEL_H 7  // Supports up to kernel height 7

template<int KERNEL_H>
__global__ void depthwise_conv2d_kernel(
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
    int stride,
    int padding,
    int dilation)
{
    extern __shared__ float s_weights[];
    
    int ow = threadIdx.x + blockIdx.x * blockDim.x;
    int oh = blockIdx.y % out_h;
    int c = blockIdx.y / out_h;
    int b = blockIdx.z;

    if (ow >= out_w || c >= channels || b >= batch) return;

    // Load weights into shared memory
    if (threadIdx.x < KERNEL_H) {
        s_weights[threadIdx.x] = weight[c * KERNEL_H + threadIdx.x];
    }
    __syncthreads();
    
    float sum = 0.0f;
    const int iw_base = ow * stride - padding;

    // Unrolled loop with boundary checks
    #pragma unroll
    for (int kh = 0; kh < KERNEL_H; ++kh) {
        int ih = oh * stride - padding + kh * dilation;
        if (ih >= 0 && ih < in_h && iw_base >= 0 && iw_base < in_w) {
            int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw_base;
            sum += input[input_idx] * s_weights[kh];
        }
    }

    // Warp-level reduction (for potential future use with accumulations)
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    sum += __ldg(&bias[c]);
    output[((b * channels + c) * out_h + oh) * out_w + ow] = sum;
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

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);
    
    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == input channels");
    }

    at::Tensor bias_val = bias.has_value() ? bias->contiguous() : at::zeros({channels}, x.options());

    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;
    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    dim3 threads(256, 1, 1);
    dim3 blocks(
        (out_w + threads.x - 1) / threads.x,
        channels * out_h,
        batch
    );

    size_t shared_mem = kernel_h * sizeof(float);
    
    // Dispatch specialized kernel based on kernel height
    switch (kernel_h) {
        case 3:
            depthwise_conv2d_kernel<3><<<blocks, threads, shared_mem>>>(
                x.data_ptr<float>(), weight.data_ptr<float>(), bias_val.data_ptr<float>(),
                output.data_ptr<float>(), batch, channels, in_h, in_w, out_h, out_w,
                stride, padding, dilation);
            break;
        case 5:
            depthwise_conv2d_kernel<5><<<blocks, threads, shared_mem>>>(
                x.data_ptr<float>(), weight.data_ptr<float>(), bias_val.data_ptr<float>(),
                output.data_ptr<float>(), batch, channels, in_h, in_w, out_h, out_w,
                stride, padding, dilation);
            break;
        case 7:
            depthwise_conv2d_kernel<7><<<blocks, threads, shared_mem>>>(
                x.data_ptr<float>(), weight.data_ptr<float>(), bias_val.data_ptr<float>(),
                output.data_ptr<float>(), batch, channels, in_h, in_w, out_h, out_w,
                stride, padding, dilation);
            break;
        default:
            throw std::invalid_argument("Unsupported kernel height");
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Conv with Shared Memory (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = c10::nullopt,
          py::arg("stride"), py::arg("padding"), py::arg("dilation"), py::arg("groups"));
}