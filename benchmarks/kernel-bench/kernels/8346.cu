#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Device function for input position calculation
__device__ inline bool compute_input_position(int out_pos, int k, int stride, int dilation, int padding, int L_in, int* in_pos) {
    *in_pos = out_pos * stride + k * dilation - padding;
    return (*in_pos >= 0) && (*in_pos < L_in);
}

// Device function for weight index calculation
__device__ inline int get_weight_index(int out_ch, int local_in_ch, int K, int group_size_in, int k) {
    return out_ch * (group_size_in * K) + local_in_ch * K + k;
}

// Device function for input index calculation
__device__ inline int get_input_index(int n, int in_ch, int L_in, int C_in, int in_pos) {
    return n * (C_in * L_in) + in_ch * L_in + in_pos;
}

__global__ void conv1d_forward_kernel_unrolled(
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
    const int group_size_in,
    const int group_size_out
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * L_out) return;

    const int out_pos = idx % L_out;
    const int out_ch = (idx / L_out) % C_out;
    const int n = idx / (L_out * C_out);
    
    const int group_idx = out_ch / group_size_out;
    float val = 0.0f;

    #pragma unroll
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        const int in_ch = group_idx * group_size_in + local_in_ch;
        
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            int in_pos;
            if (compute_input_position(out_pos, k, stride, dilation, padding, L_in, &in_pos)) {
                const float x_val = x[get_input_index(n, in_ch, L_in, C_in, in_pos)];
                const float w_val = w[get_weight_index(out_ch, local_in_ch, K, group_size_in, k)];
                val += x_val * w_val;
            }
        }
    }

    if (bias_ptr) {
        val += bias_ptr[out_ch];
    }

    y[idx] = val;
}

at::Tensor conv1d_forward_impl_unrolled(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
)
{
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");

    const int64_t N = x.size(0);
    const int64_t C_in = x.size(1);
    const int64_t L_in = x.size(2);
    const int64_t C_out = weight.size(0);
    const int64_t K = weight.size(2);
    
    const int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Invalid output length");

    auto y = torch::empty({N, C_out, L_out}, x.options());
    const float* bias_ptr = bias_opt.has_value() ? bias_opt->data_ptr<float>() : nullptr;

    const int group_size_in = C_in / groups;
    const int group_size_out = C_out / groups;

    const int total_threads = N * C_out * L_out;
    const int block_size = 256;
    const int grid_size = (total_threads + block_size - 1) / block_size;

    conv1d_forward_kernel_unrolled<<<grid_size, block_size>>>(
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
        [](at::Tensor x, at::Tensor weight, py::object bias_obj,
           int64_t stride, int64_t padding, int64_t dilation, int64_t groups) {
            c10::optional<at::Tensor> bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<at::Tensor>();
            }
            return conv1d_forward_impl_unrolled(x, weight, bias, stride, padding, dilation, groups);
        }, "Optimized 1D Convolution forward (CUDA) with loop unrolling"
    );
}
