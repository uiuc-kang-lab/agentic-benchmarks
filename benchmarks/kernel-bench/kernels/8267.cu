#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

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
    const int L_out,
    const int valid_start,
    const int valid_end
) {
    const int out_ch = blockIdx.x;
    const int out_pos = blockIdx.y * blockDim.x + threadIdx.x;
    const int n = blockIdx.z;
    
    if (out_pos >= L_out) return;

    const int group_size_out = C_out / groups;
    const int group_size_in = C_in / groups;
    const int group_idx = out_ch / group_size_out;
    const int group_start = group_idx * group_size_in;
    
    const int batch_offset = n * (C_in * L_in);
    const int weight_offset = out_ch * (group_size_in * K);
    const int output_offset = n * (C_out * L_out) + out_ch * L_out + out_pos;
    
    const int in_pos_base = out_pos * stride - padding;
    float sum = 0.0f;

    #pragma unroll 4
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        const int in_ch = group_start + local_in_ch;
        const int in_offset = batch_offset + in_ch * L_in;
        const int w_offset = weight_offset + local_in_ch * K;
        
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            const int in_pos = in_pos_base + k * dilation;
            const bool valid_pos = (in_pos >= 0) && (in_pos < L_in);
            const float x_val = valid_pos ? __ldg(&x[in_offset + in_pos]) : 0.0f;
            const float w_val = __ldg(&w[w_offset + k]);
            sum = fmaf(x_val, w_val, sum);
        }
    }

    sum += (bias_ptr != nullptr) ? __ldg(&bias_ptr[out_ch]) : 0.0f;
    y[output_offset] = sum;
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

    const int valid_start = padding;
    const int valid_end = L_in + padding;

    dim3 blockSize(256);
    dim3 gridSize(C_out, (L_out + blockSize.x - 1) / blockSize.x, N);

    conv1d_forward_kernel<<<gridSize, blockSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out, valid_start, valid_end
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
        "Divergence-free 1D Convolution forward (CUDA)"
    );
}