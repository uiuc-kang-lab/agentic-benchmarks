#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Declare constant memory for weight and bias
// Note: The total size of these arrays must not exceed the available constant memory (typically 64 KB).
extern __constant__ float const_weight[];
extern __constant__ float const_bias[];

// CUDA kernel using constant memory for weight and bias
__global__ void conv1d_forward_kernel_const(
    const float* __restrict__ x,
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
    const int group_size_out,
    const int use_bias
) {
    // Compute output indices using a 3D grid: x dim -> output position, y dim -> output channel, z dim -> batch
    int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int out_ch  = blockIdx.y * blockDim.y + threadIdx.y;
    int n       = blockIdx.z;

    if(n >= N || out_ch >= C_out || out_pos >= L_out) return;

    int group_idx = out_ch / group_size_out;
    float val = 0.0f;

    // Accumulate convolution result
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        for (int k = 0; k < K; ++k) {
            int in_pos = out_pos * stride + k * dilation - padding;
            if(in_pos >= 0 && in_pos < L_in) {
                float x_val = x[n * C_in * L_in + in_ch * L_in + in_pos];
                // Read weight from constant memory
                float w_val = const_weight[out_ch * (group_size_in * K) + local_in_ch * K + k];
                val += x_val * w_val;
            }
        }
    }
    // Add bias if used
    if(use_bias) {
        val += const_bias[out_ch];
    }
    y[n * C_out * L_out + out_ch * L_out + out_pos] = val;
}

at::Tensor conv1d_forward_impl_const(
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

    const int N = x.size(0);
    const int C_in = x.size(1);
    const int L_in = x.size(2);
    const int C_out = weight.size(0);
    const int K = weight.size(2);

    // Calculate output length
    const int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Invalid output length");

    // Create output tensor
    auto y = torch::empty({N, C_out, L_out}, x.options());

    // Group sizes for convolution
    const int group_size_in = C_in / groups;
    const int group_size_out = C_out / groups;

    // Copy weight tensor to constant memory
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), weight.nbytes(), 0, cudaMemcpyDeviceToDevice);

    int use_bias = 0;
    if(bias_opt.has_value() && bias_opt.value().defined()) {
        const at::Tensor bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias.numel() == C_out, "bias must have C_out elements");
        cudaMemcpyToSymbol(const_bias, bias.data_ptr<float>(), bias.nbytes(), 0, cudaMemcpyDeviceToDevice);
        use_bias = 1;
    }

    // Setup grid and block dimensions
    dim3 block(32, 4);  // 128 threads per block
    dim3 grid(
        (L_out + block.x - 1) / block.x,
        (C_out + block.y - 1) / block.y,
        N
    );

    // Launch the CUDA kernel
    conv1d_forward_kernel_const<<<grid, block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        L_out, group_size_in, group_size_out,
        use_bias
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](at::Tensor x, at::Tensor weight, py::object bias,
           int64_t stride, int64_t padding, int64_t dilation, int64_t groups) {
            c10::optional<at::Tensor> bias_opt;
            if (!bias.is_none()) {
                bias_opt = bias.cast<at::Tensor>();
            }
            return conv1d_forward_impl_const(x, weight, bias_opt, stride, padding, dilation, groups);
        },
        "1D Convolution forward using constant memory for weight and bias"
    );
}
