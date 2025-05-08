#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel_coalesced(
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
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int total_elements = batch * channels * out_h * out_w;
    const int total_warps = (total_elements + WARP_SIZE - 1) / WARP_SIZE;
    
    for (int warp_offset = warp_id; warp_offset < total_warps; warp_offset += gridDim.x * (blockDim.x / WARP_SIZE)) {
        const int base_idx = warp_offset * WARP_SIZE;
        const int curr_idx = base_idx + lane_id;
        
        if (curr_idx < total_elements) {
            const int ow = curr_idx % out_w;
            int tmp = curr_idx / out_w;
            const int oh = tmp % out_h;
            tmp = tmp / out_h;
            const int c = tmp % channels;
            const int n = tmp / channels;
            
            scalar_t sum = 0;
            
            const int h_base = oh * stride - padding;
            const int w_base = ow * stride - padding;
            #pragma unroll
            for (int ki = 0; ki < k; ki++) {
                #pragma unroll
                for (int kj = 0; kj < k; kj++) {
                    const int ih = h_base + ki * dilation;
                    const int iw = w_base + kj * dilation;
                    
                    if (ih >= 0 && ih < in_h && iw < in_w) {
                        const int input_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                        const int weight_idx = (c * k + ki) * k + kj;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
            
            if (bias != nullptr) {
                sum += bias[c];
            }
            
            output[curr_idx] = sum;
        }
    }
}

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel_coalesced(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int height,
    int width) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int spatial_size = height * width;
    const int total_elements = batch * out_channels * spatial_size;
    const int total_warps = (total_elements + WARP_SIZE - 1) / WARP_SIZE;
    
    for (int warp_offset = warp_id; warp_offset < total_warps; warp_offset += gridDim.x * (blockDim.x / WARP_SIZE)) {
        const int base_idx = warp_offset * WARP_SIZE;
        const int curr_idx = base_idx + lane_id;
        
        if (curr_idx < total_elements) {
            const int spatial_idx = curr_idx % spatial_size;
            const int tmp = curr_idx / spatial_size;
            const int oc = tmp % out_channels;
            const int n = tmp / out_channels;
            
            scalar_t sum = 0;
            
            for (int ic = 0; ic < in_channels; ic += 4) {
                const int vec_size = min(4, in_channels - ic);
                scalar_t input_vec[4] = {0};
                scalar_t weight_vec[4] = {0};
                
                #pragma unroll
                for (int v = 0; v < vec_size; v++) {
                    input_vec[v] = input[((n * in_channels + (ic + v)) * height * width) + spatial_idx];
                    weight_vec[v] = weight[oc * in_channels + (ic + v)];
                }
                
                #pragma unroll
                for (int v = 0; v < vec_size; v++) {
                    sum += input_vec[v] * weight_vec[v];
                }
            }
            
            if (bias != nullptr) {
                sum += bias[oc];
            }
            
            output[curr_idx] = sum;
        }
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
    
    const int total_threads = THREADS_PER_BLOCK;
    const int total_blocks = (batch * in_channels * out_h * out_w + total_threads - 1) / total_threads;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel_coalesced<scalar_t><<<total_blocks, total_threads>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            depthwise_bias.defined() ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
            depthwise_output.data_ptr<scalar_t>(),
            batch, in_channels,
            in_h, in_w,
            out_h, out_w,
            k, stride, padding, dilation);
    }));
    
    const int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
    
    const int pointwise_blocks = (batch * out_channels * out_h * out_w + total_threads - 1) / total_threads;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
        pointwise_conv2d_kernel_coalesced<scalar_t><<<pointwise_blocks, total_threads>>>(
            depthwise_output.data_ptr<scalar_t>(),
            pointwise_weight.data_ptr<scalar_t>(),
            pointwise_bias.defined() ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch,
            in_channels,
            out_channels,
            out_h, out_w);
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