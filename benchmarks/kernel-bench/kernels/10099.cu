#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define THREADS_PER_BLOCK 256

template <typename scalar_t>
__global__ void linear_indexed_depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch,
    const int channels,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int k,
    const int stride,
    const int padding,
    const int dilation) {
    
    // Calculate linear index for this thread
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch * channels * out_h * out_w;
    
    if (thread_idx >= total_elements) return;
    
    // Convert linear index to n,c,h,w coordinates
    const int w_size = out_w;
    const int h_size = out_h;
    const int c_size = channels;
    
    const int ow = thread_idx % w_size;
    const int oh = (thread_idx / w_size) % h_size;
    const int c = (thread_idx / (w_size * h_size)) % c_size;
    const int n = thread_idx / (w_size * h_size * c_size);
    
    scalar_t sum = bias ? bias[c] : scalar_t(0);
    
    #pragma unroll
    for (int ki = 0; ki < k; ki++) {
        const int ih = oh * stride - padding + ki * dilation;
        if (ih >= 0 && ih < in_h) {
            #pragma unroll
            for (int kj = 0; kj < k; kj++) {
                const int iw = ow * stride - padding + kj * dilation;
                if (iw >= 0 && iw < in_w) {
                    const int in_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                    const int w_idx = (c * k + ki) * k + kj;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    output[thread_idx] = sum;
}

template <typename scalar_t>
__global__ void linear_indexed_pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width) {
    
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch * out_channels * height * width;
    
    if (thread_idx >= total_elements) return;
    
    // Convert linear index to n,c,h,w coordinates
    const int w_size = width;
    const int h_size = height;
    const int c_size = out_channels;
    
    const int w = thread_idx % w_size;
    const int h = (thread_idx / w_size) % h_size;
    const int c = (thread_idx / (w_size * h_size)) % c_size;
    const int n = thread_idx / (w_size * h_size * c_size);
    
    scalar_t sum = bias ? bias[c] : scalar_t(0);
    
    const int spatial_offset = h * width + w;
    const int batch_offset = n * in_channels * height * width;
    
    #pragma unroll
    for (int ic = 0; ic < in_channels; ic++) {
        const int in_idx = batch_offset + ic * height * width + spatial_offset;
        const int w_idx = c * in_channels + ic;
        sum += input[in_idx] * weight[w_idx];
    }
    
    output[thread_idx] = sum;
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
    
    const int batch = x.size(0);
    const int channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int k = depthwise_weight.size(2);
    const int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    
    auto depthwise_output = torch::empty({batch, channels, out_h, out_w}, x.options());
    
    const int total_threads_depthwise = batch * channels * out_h * out_w;
    const int num_blocks_depthwise = (total_threads_depthwise + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        linear_indexed_depthwise_conv2d_kernel<scalar_t><<<num_blocks_depthwise, THREADS_PER_BLOCK>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            depthwise_bias.defined() ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
            depthwise_output.data_ptr<scalar_t>(),
            batch, channels, in_h, in_w, out_h, out_w,
            k, stride, padding, dilation);
    }));
    
    const int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
    
    const int total_threads_pointwise = batch * out_channels * out_h * out_w;
    const int num_blocks_pointwise = (total_threads_pointwise + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
        linear_indexed_pointwise_conv2d_kernel<scalar_t><<<num_blocks_pointwise, THREADS_PER_BLOCK>>>(
            depthwise_output.data_ptr<scalar_t>(),
            pointwise_weight.data_ptr<scalar_t>(),
            pointwise_bias.defined() ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch, channels, out_channels, out_h, out_w);
    }));
    
    return output;
}

at::Tensor toTensor(const py::object& obj) {
    if (obj.is_none()) return at::Tensor();
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
    auto dw = toTensor(depthwise_weight_obj);
    auto pw = toTensor(pointwise_weight_obj);
    auto db = toTensor(depthwise_bias_obj);
    auto pb = toTensor(pointwise_bias_obj);
    
    return forward_cuda(x, dw, pw, db, pb, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "Linear indexed depthwise separable convolution forward");
}