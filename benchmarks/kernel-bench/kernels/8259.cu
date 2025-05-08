#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Device function to compute input index and load value
__device__ __forceinline__ float load_input(
    const float* __restrict__ x,
    int n, int in_ch, int in_pos,
    int C_in, int L_in
) {
    if (in_pos >= 0 && in_pos < L_in) {
        return x[n * (C_in * L_in) + in_ch * L_in + in_pos];
    }
    return 0.0f;
}

// Device function to load weight value
__device__ __forceinline__ float load_weight(
    const float* __restrict__ w,
    int out_ch, int local_in_ch, int k,
    int group_size_in, int K
) {
    return w[out_ch * (group_size_in * K) + local_in_ch * K + k];
}

// Device function to handle bias addition
__device__ __forceinline__ float add_bias(
    const float* __restrict__ bias_ptr,
    float val, int out_ch
) {
    return bias_ptr ? val + bias_ptr[out_ch] : val;
}

// Device function to compute convolution for a single output element
__device__ __forceinline__ float compute_conv_element(
    const float* __restrict__ x,
    const float* __restrict__ w,
    int n, int out_ch, int out_pos,
    int C_in, int L_in, int K,
    int stride, int padding, int dilation,
    int group_size_in, int group_idx
) {
    float val = 0.0f;
    
    #pragma unroll 4
    for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            int in_pos = out_pos * stride + k * dilation - padding;
            float x_val = load_input(x, n, in_ch, in_pos, C_in, L_in);
            float w_val = load_weight(w, out_ch, local_in_ch, k, group_size_in, K);
            val += x_val * w_val;
        }
    }
    
    return val;
}

// Main kernel utilizing modular device functions
__global__ void conv1d_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr,
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
    const int L_out
) {
    int out_ch = blockIdx.x;
    int out_pos = blockIdx.y * blockDim.x + threadIdx.x;
    if (out_pos >= L_out) return;
    
    int n = blockIdx.z;

    int group_size_out = C_out / groups;
    int group_size_in = C_in / groups;
    int group_idx = out_ch / group_size_out;

    float val = compute_conv_element(
        x, w, n, out_ch, out_pos,
        C_in, L_in, K, stride, padding, dilation,
        group_size_in, group_idx
    );

    val = add_bias(bias_ptr, val, out_ch);
    y[n * (C_out * L_out) + out_ch * L_out + out_pos] = val;
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
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    auto x_sizes = x.sizes();
    int64_t N    = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K     = w_sizes[2];

    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    dim3 blockSize(256);
    dim3 gridSize(C_out, (L_out + blockSize.x - 1) / blockSize.x, N);

    conv1d_forward_kernel<<<gridSize, blockSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel failed: ", cudaGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](at::Tensor x,
           at::Tensor weight,
           py::object bias_obj,
           int64_t stride,
           int64_t padding,
           int64_t dilation,
           int64_t groups) {
            c10::optional<at::Tensor> bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<at::Tensor>();
            }
            return conv1d_forward_impl(x, weight, bias, stride, padding, dilation, groups);
        },
        "Modular optimized 1D Convolution forward (CUDA)"
    );
}