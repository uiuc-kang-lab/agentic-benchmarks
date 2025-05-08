#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// This kernel refactors the inner loop to precompute the valid range of kernel indices (k_start and k_end) using branchless arithmetic.
// This eliminates per-iteration conditional checks and minimizes warp divergence within the inner loops.
__global__ void conv1d_forward_kernel_branchless(
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
    // Each thread computes one output element identified by (n, out_ch, out_pos)
    int out_ch = blockIdx.x;
    int out_pos = blockIdx.y * blockDim.x + threadIdx.x;
    if (out_pos >= L_out) return;
    int n = blockIdx.z;

    // Determine channel grouping
    int group_size_out = C_out / groups;
    int group_size_in  = C_in  / groups;
    int group_idx = out_ch / group_size_out;

    // Compute starting input index offset
    int LHS = out_pos * stride - padding;

    // Compute k_start in branchless fashion: if (LHS < 0) then ceil(-LHS/dilation) else 0
    int k_start = ((-LHS + dilation - 1) / dilation) * (int)(LHS < 0);

    // Compute k_end in branchless fashion: maximum k such that LHS + k*dilation is in bounds
    int tmp = L_in - LHS;
    int k_end = ((((tmp - 1) / dilation) + 1) * (int)(tmp > 0));
    k_end = k_end < K ? k_end : K;

    float sum = 0.0f;
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        // Loop only over the valid k indices, eliminating branch divergence within the loop
        for (int k = k_start; k < k_end; ++k) {
            int in_pos = LHS + k * dilation;  // Guaranteed to be in [0, L_in) by our k bounds
            float x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
            float w_val = w[out_ch * (group_size_in * K) + local_in_ch * K + k];
            sum += x_val * w_val;
        }
    }

    if (bias_ptr) {
        sum += bias_ptr[out_ch];
    }

    y[n * (C_out * L_out) + out_ch * L_out + out_pos] = sum;
}

// Host function to configure and launch the CUDA kernel
at::Tensor conv1d_forward_impl_branchless(
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

    // Configure grid and block dimensions
    dim3 blockSize(256);
    dim3 gridSize(C_out, (L_out + blockSize.x - 1) / blockSize.x, N);

    conv1d_forward_kernel_branchless<<<gridSize, blockSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups, (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_branchless failed: ", cudaGetErrorString(err));

    return y;
}

// Pybind11 binding
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
            return conv1d_forward_impl_branchless(x, weight, bias, stride, padding, dilation, groups);
        },
        "Optimized 1D Convolution forward (CUDA) with minimized warp divergence by refactoring conditional logic"
    );
}
