#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define BLOCK_SIZE 256
#define STRIDE_FACTOR 4

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

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int total_elements = batch * channels * out_h * out_w;
    const int elements_per_block = BLOCK_SIZE * STRIDE_FACTOR;
    const int start_idx = bid * elements_per_block + tid;
    
    #pragma unroll
    for (int offset = 0; offset < STRIDE_FACTOR; offset++) {
        const int idx = start_idx + offset * BLOCK_SIZE;
        if (idx >= total_elements) continue;

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
                    sum += input[n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw] *
                           weight[c * k * k + i * k + j];
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[c];
        }
        output[idx] = sum;
    }
}

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width) {
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Each thread processes multiple elements in x direction
    for (int x_offset = 0; x_offset < STRIDE_FACTOR; x_offset++) {
        const int out_x = (bx * BLOCK_DIM_X + tx) * STRIDE_FACTOR + x_offset;
        if (out_x >= width) continue;
        
        const int out_y = by * blockDim.y + ty;
        if (out_y >= height) continue;

        const int batch_idx = blockIdx.z / out_channels;
        const int out_ch = blockIdx.z % out_channels;
        
        if (batch_idx >= batch) continue;

        scalar_t sum = 0;
        const int spatial_offset = out_y * width + out_x;
        
        #pragma unroll 4
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            sum += input[batch_idx * in_channels * height * width + 
                        in_ch * height * width + spatial_offset] *
                   weight[out_ch * in_channels + in_ch];
        }
        
        if (bias != nullptr) {
            sum += bias[out_ch];
        }
        
        output[batch_idx * out_channels * height * width +
               out_ch * height * width +
               spatial_offset] = sum;
    }
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

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int k = depthwise_weight.size(2);
    const int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

    auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

    const int total_elements = batch * in_channels * out_h * out_w;
    const int blocks_dw = (total_elements + (BLOCK_SIZE * STRIDE_FACTOR) - 1) / (BLOCK_SIZE * STRIDE_FACTOR);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks_dw, BLOCK_SIZE>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            depthwise_bias.defined() ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
            depthwise_output.data_ptr<scalar_t>(),
            batch, in_channels, in_h, in_w, out_h, out_w,
            k, stride, padding, dilation);
    }));

    const int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (out_w + threads.x * STRIDE_FACTOR - 1) / (threads.x * STRIDE_FACTOR),
        (out_h + threads.y - 1) / threads.y,
        batch * out_channels
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
        pointwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            depthwise_output.data_ptr<scalar_t>(),
            pointwise_weight.data_ptr<scalar_t>(),
            pointwise_bias.defined() ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch, in_channels, out_channels, out_h, out_w);
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

    return forward_cuda(
        toTensor(x_obj),
        toTensor(depthwise_weight_obj),
        toTensor(pointwise_weight_obj),
        toTensor(depthwise_bias_obj),
        toTensor(pointwise_bias_obj),
        stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward");
}