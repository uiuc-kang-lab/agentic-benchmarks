#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace py = pybind11;

// Device function to compute output indices
__device__ __forceinline__ void compute_output_indices(
    int index,
    int out_w,
    int out_h,
    int out_channels,
    int& n,
    int& oc,
    int& out_y,
    int& out_x
) {
    out_x = index % out_w;
    int tmp = index / out_w;
    out_y = tmp % out_h;
    tmp = tmp / out_h;
    oc = tmp % out_channels;
    n = tmp / out_channels;
}

// Device function to check if input coordinates are valid
__device__ __forceinline__ bool is_valid_input_coord(
    int in_y,
    int in_x,
    int in_h,
    int in_w
) {
    return (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w);
}

// Device function to compute input and weight indices
__device__ __forceinline__ void compute_indices(
    int n,
    int ic,
    int in_y,
    int in_x,
    int oc,
    int ky,
    int kx,
    int in_h,
    int in_w,
    int kernel_w,
    int& input_idx,
    int& weight_idx
) {
    int input_channel_stride = in_h * in_w;
    int weight_kernel_stride = kernel_w;
    
    input_idx = ((n * ic) * in_h + in_y) * in_w + in_x;
    weight_idx = ((ic * oc) * kernel_w + ky) * kernel_w + kx;
}

// Device function to compute single output element
__device__ __forceinline__ float compute_output_element(
    const float* input,
    const float* weight,
    int n,
    int oc,
    int out_y,
    int out_x,
    int in_channels,
    int in_h,
    int in_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    float sum = 0.0f;
    
    int base_y = out_y + pad_h;
    int base_x = out_x + pad_w;

    #pragma unroll 4
    for (int ky = 0; ky < kernel_h; ky++) {
        int t_y = base_y - ky;
        if (t_y % stride_h != 0) continue;
        int in_y = t_y / stride_h;
        if (!is_valid_input_coord(in_y, 0, in_h, in_w)) continue;

        #pragma unroll 4
        for (int kx = 0; kx < kernel_w; kx++) {
            int t_x = base_x - kx;
            if (t_x % stride_w != 0) continue;
            int in_x = t_x / stride_w;
            if (!is_valid_input_coord(in_x, 0, 0, in_w)) continue;

            #pragma unroll 4
            for (int ic = 0; ic < in_channels; ic++) {
                int input_idx, weight_idx;
                compute_indices(n, ic, in_y, in_x, oc, ky, kx, in_h, in_w, kernel_w, input_idx, weight_idx);
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    return sum;
}

__global__ void conv_transpose2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int out_h,
    int out_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    bool has_bias
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_channels * out_h * out_w;
    if (index >= total) return;

    int n, oc, out_y, out_x;
    compute_output_indices(index, out_w, out_h, out_channels, n, oc, out_y, out_x);
    
    float sum = compute_output_element(
        input, weight, n, oc, out_y, out_x,
        in_channels, in_h, in_w, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w
    );
    
    if (has_bias) {
        sum += bias[oc];
    }
    
    output[index] = sum;
}

torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    auto N = input.size(0);
    auto in_channels = input.size(1);
    auto in_h = input.size(2);
    auto in_w = input.size(3);
    auto out_channels = weight.size(1);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];
    
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w;
    
    auto output = torch::zeros({N, out_channels, out_h, out_w}, input.options());
    
    int total = N * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    bool has_bias = bias.has_value();
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;
    
    conv_transpose2d_forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, in_channels, in_h, in_w,
        out_channels, kernel_h, kernel_w,
        out_h, out_w, stride_h, stride_w,
        pad_h, pad_w, has_bias
    );
    
    return output;
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    c10::optional<torch::Tensor> bias = c10::nullopt;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
    }
    return conv_transpose2d_forward_cuda(input, weight, bias, stride, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward (modular approach)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}