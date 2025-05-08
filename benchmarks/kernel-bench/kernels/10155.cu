#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k, int stride, int padding, int dilation) {
    
    // Each thread processes one output pixel across all channels
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int pixels_per_block = blockDim.x;
    const int pixel_idx = bid * pixels_per_block + tid;
    
    if (pixel_idx >= batch * out_h * out_w)
        return;
    
    const int n = pixel_idx / (out_h * out_w);
    const int out_pos = pixel_idx % (out_h * out_w);
    const int oh = out_pos / out_w;
    const int ow = out_pos % out_w;
    
    // Process all channels for this pixel position
    for (int c = 0; c < channels; c++) {
        scalar_t sum = 0;
        
        #pragma unroll
        for (int i = 0; i < k; i++) {
            const int ih = oh * stride - padding + i * dilation;
            
            #pragma unroll
            for (int j = 0; j < k; j++) {
                const int iw = ow * stride - padding + j * dilation;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    const int in_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                    const int w_idx = (c * k + i) * k + j;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
        
        if (bias != nullptr)
            sum += bias[c];
            
        const int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
        output[out_idx] = sum;
    }
}

template <typename scalar_t>
__global__ void pointwise_conv2d_coalesced_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch, int in_channels, int out_channels,
    int height, int width) {
    
    // Each thread block handles a tile of output pixels
    const int tid = threadIdx.x;
    const int tile_idx = blockIdx.x;
    
    // Calculate spatial position
    const int pixels_per_block = blockDim.x / WARP_SIZE;
    const int pixel_offset = tile_idx * pixels_per_block;
    const int pixel_id = pixel_offset + (tid / WARP_SIZE);
    
    if (pixel_id >= batch * height * width)
        return;
    
    // Calculate batch and spatial indices
    const int n = pixel_id / (height * width);
    const int hw = pixel_id % (height * width);
    const int h = hw / width;
    const int w = hw % width;
    
    // Each thread within a warp handles different output channels
    const int lane_id = tid % WARP_SIZE;
    
    // Process output channels in chunks of WARP_SIZE
    for (int oc_base = 0; oc_base < out_channels; oc_base += WARP_SIZE) {
        const int oc = oc_base + lane_id;
        if (oc < out_channels) {
            scalar_t sum = 0;
            
            // Coalesced read of input channels
            for (int ic = 0; ic < in_channels; ic++) {
                const scalar_t in_val = input[((n * in_channels + ic) * height + h) * width + w];
                const scalar_t w_val = weight[oc * in_channels + ic];
                sum += in_val * w_val;
            }
            
            if (bias != nullptr)
                sum += bias[oc];
                
            // Coalesced write to output
            output[((n * out_channels + oc) * height + h) * width + w] = sum;
        }
    }
}

torch::Tensor forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride, int padding, int dilation) {
    
    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int k = depthwise_weight.size(2);
    const int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    
    auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());
    
    // Launch depthwise kernel
    const int total_pixels = batch * out_h * out_w;
    const dim3 blocks_dw((total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const dim3 threads_dw(BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks_dw, threads_dw>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            depthwise_bias.defined() ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
            depthwise_output.data_ptr<scalar_t>(),
            batch, in_channels, in_h, in_w, out_h, out_w,
            k, stride, padding, dilation);
    }));
    
    // Pointwise convolution
    const int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
    
    // Configure launch parameters for coalesced pointwise kernel
    const int pixels_per_block = BLOCK_SIZE / WARP_SIZE;
    const int total_blocks = (total_pixels + pixels_per_block - 1) / pixels_per_block;
    const dim3 blocks_pw(total_blocks);
    const dim3 threads_pw(BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
        pointwise_conv2d_coalesced_kernel<scalar_t><<<blocks_pw, threads_pw>>>(
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
    int stride, int padding, int dilation) {
    
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
    m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward (coalesced)");
}