#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Reuse common indexing calculations for efficiency
__device__ inline bool compute_input_position(int out_pos, int k, int stride, int dilation, int padding, int L_in, int* in_pos) {
    *in_pos = out_pos * stride + k * dilation - padding;
    return (*in_pos >= 0) && (*in_pos < L_in);
}

__device__ inline int get_weight_index(int out_ch, int local_in_ch, int K, int group_size_in, int k) {
    return out_ch * (group_size_in * K) + local_in_ch * K + k;
}

// Kernel to reduce warp divergence with loop unrolling
__global__ void conv1d_forward_kernel_unrolled(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    const int N,
    const int C_in,
    const int L_in,
    const int C_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out,
    const int group_size_in,
    const int group_size_out
) {
    // Compute thread indices
    const int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_ch = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;

    if (n >= N || out_ch >= C_out || out_pos >= L_out) return;

    const int group_idx = out_ch / group_size_out;
    float val = 0.0f;

    #pragma unroll 4
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        const int in_ch = group_idx * group_size_in + local_in_ch;

        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            int in_pos = out_pos * stride + k * dilation - padding;
            const float x_val = (in_pos >= 0 && in_pos < L_in) ? x[n * C_in * L_in + in_ch * L_in + in_pos] : 0.0f;
            const float w_val = w[out_ch * (group_size_in * K) + local_in_ch * K + k];
            val += x_val * w_val;
        }
    }

    if (bias) val += bias[out_ch];

    y[n * C_out * L_out + out_ch * L_out + out_pos] = val;
}

at::Tensor conv1d_forward_impl_unrolled(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

    const int64_t N = x.size(0);
    const int64_t C_in = x.size(1);
    const int64_t L_in = x.size(2);
    const int64_t C_out = weight.size(0);
    const int64_t K = weight.size(2);

    const int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    auto y = torch::empty({N, C_out, L_out}, x.options());

    const int group_size_in = C_in / groups;
    const int group_size_out = C_out / groups;

    const float* bias_ptr = bias_opt.has_value() ? bias_opt->data_ptr<float>() : nullptr;

    // Configure grid and block dimensions
    dim3 block(32, 4);  // 128 threads/block
    dim3 grid((L_out + block.x - 1) / block.x, (C_out + block.y - 1) / block.y, N);

    // Launch the optimized kernel
    conv1d_forward_kernel_unrolled<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        (int)stride, (int)padding, (int)dilation, (int)groups, L_out,
        group_size_in, group_size_out
    );

    // Validate the launch operation
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
        [](at::Tensor x, at::Tensor weight, py::object bias,
           int64_t stride, int64_t padding, int64_t dilation, int64_t groups) {
            return conv1d_forward_impl_unrolled(x, weight,
                bias.is_none() ? c10::nullopt : c10::optional<at::Tensor>(bias.cast<at::Tensor>()),
                stride, padding, dilation, groups);
        }, "1D Convolution optimized with loop unrolling");
}
