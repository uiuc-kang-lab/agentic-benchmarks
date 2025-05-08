#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Optimized for H100 GPU - adjusted block dimensions for better occupancy
#define BLOCK_X 32
#define BLOCK_Y 4
#define BLOCK_Z 4

template <typename scalar_t>
__global__ void optimized_depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {
    
    // Direct 3D thread indexing
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // width
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // height
    const int z = blockIdx.z * blockDim.z + threadIdx.z; // batch * channels
    
    if (x >= out_w || y >= out_h || z >= batch_size * channels)
        return;
        
    const int b = z / channels;
    const int c = z % channels;
    
    scalar_t sum = 0;
    
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int h_in = y * stride - padding + kh * dilation;
        if (h_in >= 0 && h_in < in_h) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                const int w_in = x * stride - padding + kw * dilation;
                if (w_in >= 0 && w_in < in_w) {
                    const int in_idx = ((b * channels + c) * in_h + h_in) * in_w + w_in;
                    const int w_idx = (c * kernel_size + kh) * kernel_size + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[c];
    }
    
    const int out_idx = ((b * channels + c) * out_h + y) * out_w + x;
    output[out_idx] = sum;
}

template <typename scalar_t>
__global__ void optimized_pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width) {
    
    // Direct 3D thread indexing
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // width
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // height
    const int z = blockIdx.z * blockDim.z + threadIdx.z; // batch * out_channels
    
    if (x >= width || y >= height || z >= batch_size * out_channels)
        return;
        
    const int b = z / out_channels;
    const int oc = z % out_channels;
    
    scalar_t sum = 0;
    
    #pragma unroll 4
    for (int ic = 0; ic < in_channels; ic++) {
        const int in_idx = ((b * in_channels + ic) * height + y) * width + x;
        const int w_idx = oc * in_channels + ic;
        sum += input[in_idx] * weight[w_idx];
    }
    
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    const int out_idx = ((b * out_channels + oc) * height + y) * width + x;
    output[out_idx] = sum;
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

    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    const int batch_size = x.size(0);
    const int channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_size = depthwise_weight.size(2);
    const int out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto depthwise_output = torch::empty({batch_size, channels, out_h, out_w}, x.options());
    
    // 3D grid configuration for depthwise conv
    dim3 threads(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 blocks(
        (out_w + threads.x - 1) / threads.x,
        (out_h + threads.y - 1) / threads.y,
        (batch_size * channels + threads.z - 1) / threads.z
    );
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_depthwise_conv2d_cuda", ([&] {
        optimized_depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            depthwise_bias.defined() ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
            depthwise_output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            in_h, in_w,
            out_h, out_w,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));
    
    const int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());
    
    // 3D grid configuration for pointwise conv
    dim3 pw_blocks(
        (out_w + threads.x - 1) / threads.x,
        (out_h + threads.y - 1) / threads.y,
        (batch_size * out_channels + threads.z - 1) / threads.z
    );
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_pointwise_conv2d_cuda", ([&] {
        optimized_pointwise_conv2d_kernel<scalar_t><<<pw_blocks, threads>>>(
            depthwise_output.data_ptr<scalar_t>(),
            pointwise_weight.data_ptr<scalar_t>(),
            pointwise_bias.defined() ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            out_channels,
            out_h,
            out_w
        );
    }));
    
    return output;
}

at::Tensor toTensor(const py::object& obj) {
    if (obj.is_none()) {
        return at::Tensor();
    }
    try {
        return obj.cast<at::Tensor>();
    } catch (const py::cast_error& e) {
        if (py::hasattr(obj, "data")) {
            return obj.attr("data").cast<at::Tensor>();
        }
        throw std::runtime_error("Expected a torch Tensor or Parameter.");
    }
}

at::Tensor forward_wrapper(
    py::object x_obj,
    py::object depthwise_weight_obj,
    py::object pointwise_weight_obj,
    py::object depthwise_bias_obj,
    py::object pointwise_bias_obj,
    int stride,
    int padding,
    int dilation) {
    
    auto x = toTensor(x_obj);
    auto depthwise_weight = toTensor(depthwise_weight_obj);
    auto pointwise_weight = toTensor(pointwise_weight_obj);
    auto depthwise_bias = toTensor(depthwise_bias_obj);
    auto pointwise_bias = toTensor(pointwise_bias_obj);
    
    return forward_cuda(
        x, depthwise_weight, pointwise_weight,
        depthwise_bias, pointwise_bias,
        stride, padding, dilation
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "Optimized CUDA depthwise separable convolution forward");
}