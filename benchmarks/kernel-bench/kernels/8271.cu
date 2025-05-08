#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Template kernel for different block sizes
template<int BLOCK_SIZE>
__global__ void conv1d_forward_kernel_templated(
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
    int out_pos = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    if (out_pos >= L_out) return;
    int n = blockIdx.z;

    int group_size_out = C_out / groups;
    int group_size_in = C_in / groups;
    int group_idx = out_ch / group_size_out;

    float sum = 0.0f;
    
    #pragma unroll 4
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            int in_pos = out_pos * stride + k * dilation - padding;
            if (in_pos >= 0 && in_pos < L_in) {
                float x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                float w_val = w[out_ch * (group_size_in * K) + local_in_ch * K + k];
                sum += x_val * w_val;
            }
        }
    }

    if (bias_ptr) {
        sum += bias_ptr[out_ch];
    }

    y[n * (C_out * L_out) + out_ch * L_out + out_pos] = sum;
}

// Helper function to select optimal block size based on input parameters
int select_block_size(int L_out, int C_out) {
    if (L_out <= 32) return 32;
    if (L_out <= 64) return 64;
    if (L_out <= 128) return 128;
    if (C_out >= 512 && L_out <= 256) return 256;
    return 512;
}

// Launch wrapper that selects appropriate kernel based on block size
void launch_kernel_with_block_size(
    const float* x,
    const float* w,
    const float* bias_ptr,
    float* y,
    int N, int C_in, int L_in, int C_out, int K,
    int stride, int padding, int dilation, int groups,
    int L_out, int block_size
) {
    dim3 grid(C_out, (L_out + block_size - 1) / block_size, N);

    switch(block_size) {
        case 32:
            conv1d_forward_kernel_templated<32><<<grid, 32>>>(
                x, w, bias_ptr, y, N, C_in, L_in, C_out, K,
                stride, padding, dilation, groups, L_out);
            break;
        case 64:
            conv1d_forward_kernel_templated<64><<<grid, 64>>>(
                x, w, bias_ptr, y, N, C_in, L_in, C_out, K,
                stride, padding, dilation, groups, L_out);
            break;
        case 128:
            conv1d_forward_kernel_templated<128><<<grid, 128>>>(
                x, w, bias_ptr, y, N, C_in, L_in, C_out, K,
                stride, padding, dilation, groups, L_out);
            break;
        case 256:
            conv1d_forward_kernel_templated<256><<<grid, 256>>>(
                x, w, bias_ptr, y, N, C_in, L_in, C_out, K,
                stride, padding, dilation, groups, L_out);
            break;
        default:
            conv1d_forward_kernel_templated<512><<<grid, 512>>>(
                x, w, bias_ptr, y, N, C_in, L_in, C_out, K,
                stride, padding, dilation, groups, L_out);
    }
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

    int block_size = select_block_size(L_out, C_out);

    launch_kernel_with_block_size(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        stride, padding, dilation, groups,
        L_out, block_size
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
        "Dynamic block-size tuned 1D Convolution forward (CUDA)"
    );
}