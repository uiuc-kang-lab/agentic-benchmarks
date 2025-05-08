#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Define thread block sizes for depthwise and pointwise kernels
#define DW_THREADS 256
#define PW_BLOCK_DIM_X 32

// Depthwise convolution kernel using 1D linear indexing.
// Input: [batch, channels, in_h, in_w]
// Weight: [channels, 1, k, k]
// Output: [batch, channels, out_h, out_w]
// Each thread computes one output element.

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,   // can be nullptr
    scalar_t* __restrict__ output,
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

    int idx = blockIdx.x * DW_THREADS + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    // Decode the linear index
    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    tmp = tmp / out_h;
    int c = tmp % channels;
    int n = tmp / channels;

    scalar_t sum = 0;
    // Iterate over kernel window
    #pragma unroll
    for (int i = 0; i < k; ++i) {
        int ih = oh * stride - padding + i * dilation;
        #pragma unroll
        for (int j = 0; j < k; ++j) {
            int iw = ow * stride - padding + j * dilation;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int input_idx = n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
                int weight_idx = c * k * k + i * k + j;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    if (bias != nullptr) {
        sum += bias[c];
    }
    output[idx] = sum;
}

// Pointwise convolution kernel optimized for memory coalescing.
// Input: [batch, in_channels, height, width]
// Weight: [out_channels, in_channels] (from original pointwise weight [out_channels, in_channels, 1, 1])
// Output: [batch, out_channels, height, width]
// Each thread computes one output pixel; grid is set so that threads along x dimension are consecutive.

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,   // can be nullptr
    scalar_t* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int height,
    int width) {

    // Compute coordinates: each thread computes one output pixel for given (batch, out_channel)
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;  // along width
    int out_y = blockIdx.y;  // along height
    int idx_z = blockIdx.z;  // encodes batch and out_channel
    int batch_idx = idx_z / out_channels;
    int out_ch = idx_z % out_channels;

    if (out_x >= width || out_y >= height) return;

    scalar_t sum = 0;
    int spatial_offset = out_y * width + out_x;
    int input_batch_offset = batch_idx * in_channels * height * width;
    int weight_offset = out_ch * in_channels;

    // Sum over input channels (in_channels dimension)
    for (int ic = 0; ic < in_channels; ++ic) {
        int input_idx = input_batch_offset + ic * height * width + spatial_offset;
        sum += input[input_idx] * weight[weight_offset + ic];
    }

    if (bias != nullptr) {
        sum += bias[out_ch];
    }

    int output_idx = batch_idx * out_channels * height * width + out_ch * height * width + spatial_offset;
    output[output_idx] = sum;
}


// CUDA forward function that launches the depthwise and pointwise convolution kernels.

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
    TORCH_CHECK(depthwise_weight.is_cuda(), "Depthwise weight must be a CUDA tensor");
    TORCH_CHECK(pointwise_weight.is_cuda(), "Pointwise weight must be a CUDA tensor");
    if (depthwise_bias.defined() && depthwise_bias.numel() > 0)
        TORCH_CHECK(depthwise_bias.is_cuda(), "Depthwise bias must be a CUDA tensor if provided");
    if (pointwise_bias.defined() && pointwise_bias.numel() > 0)
        TORCH_CHECK(pointwise_bias.is_cuda(), "Pointwise bias must be a CUDA tensor if provided");

    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    // Depthwise convolution parameters
    int k = depthwise_weight.size(2); // weight shape: [in_channels, 1, k, k]
    int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

    auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

    int total_dw = batch * in_channels * out_h * out_w;
    int blocks_dw = (total_dw + DW_THREADS - 1) / DW_THREADS;

    const void* dw_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0) ? depthwise_bias.data_ptr() : nullptr;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks_dw, DW_THREADS>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            reinterpret_cast<const scalar_t*>(dw_bias_ptr),
            depthwise_output.data_ptr<scalar_t>(),
            batch, in_channels, in_h, in_w,
            out_h, out_w,
            k, stride, padding, dilation);
    }));

    // Pointwise convolution parameters
    int out_channels = pointwise_weight.size(0); // weight shape: [out_channels, in_channels, 1, 1] -> treated as [out_channels, in_channels]
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

    // Configure grid: each thread computes one output pixel. We use 1D block (PW_BLOCK_DIM_X threads) along width.
    dim3 pt_block(PW_BLOCK_DIM_X, 1, 1);
    dim3 pt_grid((out_w + PW_BLOCK_DIM_X - 1) / PW_BLOCK_DIM_X, out_h, batch * out_channels);

    const void* pt_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0) ? pointwise_bias.data_ptr() : nullptr;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
        pointwise_conv2d_kernel<scalar_t><<<pt_grid, pt_block>>>(
            depthwise_output.data_ptr<scalar_t>(),
            reinterpret_cast<const scalar_t*>(pointwise_weight.data_ptr<scalar_t>()),
            reinterpret_cast<const scalar_t*>(pt_bias_ptr),
            output.data_ptr<scalar_t>(),
            batch, in_channels, out_channels,
            out_h, out_w);
    }));

    return output;
}

// Helper function to convert a py::object to an at::Tensor
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

// Wrapper function for Python interface: forward(tensor, tensor, tensor, tensor, tensor, int, int, int) -> tensor
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

    return forward_cuda(x, depthwise_weight, pointwise_weight,
                        depthwise_bias, pointwise_bias,
                        stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward");
}
