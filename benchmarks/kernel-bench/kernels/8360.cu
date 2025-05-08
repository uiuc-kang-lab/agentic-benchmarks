#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

__device__ inline int compute_inpos(int outpos, int k, int stride, int dilation, int padding) {
    return outpos * stride + k * dilation - padding;
}

__global__ void conv1d_forward_kernel(
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
    const int gsize_in,
    const int gsize_out
) {
    // Optimized 2D grid mapping
    const int outpos = blockIdx.x * blockDim.x + threadIdx.x;
    const int outc = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;
    
    if (n >= N || outc >= C_out || outpos >= L_out) return;
    
    const int group_idx = outc / gsize_out;
    float val = 0.0f;

    #pragma unroll 4
    for (int l_inc = 0; l_inc < gsize_in; ++l_inc) {
        const int in_ch = group_idx * gsize_in + l_inc;
        
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            int inpos = compute_inpos(outpos, k, stride, dilation, padding);
            const bool valid = (inpos >= 0) && (inpos < L_in);
            const float xval = valid ? __ldg(&x[n*C_in*L_in + in_ch*L_in + inpos]) : 0.0f;
            val += xval * __ldg(&w[outc*(gsize_in*K) + l_inc*K + k]);
        }
    }

    if (bias) val += __ldg(&bias[outc]);
    y[n*C_out*L_out + outc*L_out + outpos] = val;
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
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int L_in = x.size(2);
    const int C_out = weight.size(0);
    const int K = weight.size(2);
    
    const int L_out = (L_in + 2*padding - dilation*(K-1) -1)/stride + 1;
    TORCH_CHECK(L_out > 0, "Invalid output length");
    
    auto y = torch::empty({N, C_out, L_out}, x.options());
    const float* bias_ptr = bias_opt.has_value() ? bias_opt->data_ptr<float>() : nullptr;
    
    const int gsize_in = C_in/groups;
    const int gsize_out = C_out/groups;
    
    dim3 block(16, 8);  // Adjusted block size for better occupancy
    dim3 grid(
        (L_out + block.x-1) / block.x,
        (C_out + block.y-1) / block.y,
        N
    );
    
    conv1d_forward_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        stride, padding, dilation, groups, L_out,
        gsize_in, gsize_out
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
        }, "Optimized 1D Conv with tuned 2D grid"
    );
}
