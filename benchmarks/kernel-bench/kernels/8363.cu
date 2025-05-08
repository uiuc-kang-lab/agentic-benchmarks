#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Specialized device functions for different components
__device__ __forceinline__ int get_input_index(int n, int in_ch, int pos, int C_in, int L_in) {
    return n * (C_in * L_in) + in_ch * L_in + pos;
}

__device__ __forceinline__ int get_weight_index(int out_ch, int in_ch, int k, int group_size_in, int K) {
    return out_ch * (group_size_in * K) + in_ch * K + k;
}

__device__ __forceinline__ int get_output_index(int n, int out_ch, int pos, int C_out, int L_out) {
    return n * (C_out * L_out) + out_ch * L_out + pos;
}

__device__ __forceinline__ bool is_valid_position(int pos, int length) {
    return pos >= 0 && pos < length;
}

__device__ __forceinline__ float compute_conv_element(
    const float* __restrict__ x,
    const float* __restrict__ w,
    int n, int out_ch, int in_ch,
    int out_pos, int K, int stride, int dilation, int padding,
    int C_in, int L_in, int group_size_in
) {
    float result = 0.0f;
    
    #pragma unroll 4
    for (int k = 0; k < K; k++) {
        const int in_pos = out_pos * stride + k * dilation - padding;
        const bool valid = is_valid_position(in_pos, L_in);
        const float x_val = valid ? __ldg(&x[get_input_index(n, in_ch, in_pos, C_in, L_in)]) : 0.0f;
        const float w_val = __ldg(&w[get_weight_index(out_ch, in_ch, k, group_size_in, K)]);
        result += x_val * w_val;
    }
    return result;
}

__global__ void conv1d_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    const int N, const int C_in, const int L_in,
    const int C_out, const int K,
    const int stride, const int padding, const int dilation,
    const int groups, const int L_out,
    const int group_size_in, const int group_size_out
) {
    const int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_ch = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;

    if (n >= N || out_ch >= C_out || out_pos >= L_out) return;

    const int group_idx = out_ch / group_size_out;
    float val = 0.0f;

    #pragma unroll 2
    for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
        const int in_ch = group_idx * group_size_in + local_in_ch;
        val += compute_conv_element(
            x, w, n, out_ch, local_in_ch,
            out_pos, K, stride, dilation, padding,
            C_in, L_in, group_size_in
        );
    }

    if (bias) {
        val += __ldg(&bias[out_ch]);
    }

    y[get_output_index(n, out_ch, out_pos, C_out, L_out)] = val;
}

at::Tensor conv1d_forward_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Input tensors must be CUDA tensors");
    
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int L_in = x.size(2);
    const int C_out = weight.size(0);
    const int K = weight.size(2);
    
    const int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Invalid output length");
    
    auto y = torch::empty({N, C_out, L_out}, x.options());
    const float* bias_ptr = bias_opt.has_value() ? bias_opt->data_ptr<float>() : nullptr;
    
    const int group_size_in = C_in / groups;
    const int group_size_out = C_out / groups;
    
    dim3 block(32, 4);
    dim3 grid(
        (L_out + block.x - 1) / block.x,
        (C_out + block.y - 1) / block.y,
        N
    );
    
    conv1d_forward_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        stride, padding, dilation, groups, L_out,
        group_size_in, group_size_out
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
        [](at::Tensor x, at::Tensor weight, py::object bias,
           int64_t stride, int64_t padding, int64_t dilation, int64_t groups) {
            return conv1d_forward_impl(x, weight,
                bias.is_none() ? c10::nullopt : c10::optional<at::Tensor>(bias.cast<at::Tensor>()),
                stride, padding, dilation, groups);
        },
        "Modular optimized 1D convolution forward"
    );
}