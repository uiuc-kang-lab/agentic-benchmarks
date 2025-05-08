#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Kernel implementing 1D convolution with grid-stride loops
__global__ void conv1d_forward_kernel_gridstride(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr,
    float* __restrict__ y,
    const int N,         // batch size
    const int C_in,      // number of input channels
    const int L_in,      // input length
    const int C_out,     // number of output channels
    const int K,         // kernel size
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out      // output length
) {
    // Total number of output elements
    int total = N * C_out * L_out;
    
    // Grid-stride loop: each thread processes multiple output elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        // Compute output coordinates (n, out_ch, out_pos) from flattened index
        int out_pos = idx % L_out;
        int tmp = idx / L_out;
        int out_ch = tmp % C_out;
        int n = tmp / C_out;

        // Determine channel group information
        int group_size_out = C_out / groups;
        int group_size_in = C_in / groups;
        int group_idx = out_ch / group_size_out;

        float sum = 0.0f;
        int base_in = group_idx * group_size_in;
        
        // Loop over the input channels within the group
        for (int local_in = 0; local_in < group_size_in; ++local_in) {
            int in_ch = base_in + local_in;
            
            // Loop over the kernel elements
            for (int k = 0; k < K; ++k) {
                int in_pos = out_pos * stride + k * dilation - padding;
                if (in_pos >= 0 && in_pos < L_in) {
                    int x_index = n * (C_in * L_in) + in_ch * L_in + in_pos;
                    int w_index = out_ch * (group_size_in * K) + local_in * K + k;
                    sum += x[x_index] * w[w_index];
                }
            }
        }
        
        // Add bias if provided
        if (bias_ptr) {
            sum += bias_ptr[out_ch];
        }

        int y_index = n * (C_out * L_out) + out_ch * L_out + out_pos;
        y[y_index] = sum;
    }
}

// Host function that sets up dimensions and launches the grid-stride kernel
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

    // Input dimensions: [N, C_in, L_in]
    auto x_sizes = x.sizes();
    int N = x_sizes[0];
    int C_in = x_sizes[1];
    int L_in = x_sizes[2];

    // Weight dimensions: [C_out, C_in/groups, K]
    auto w_sizes = weight.sizes();
    int C_out = w_sizes[0];
    int K = w_sizes[2];

    // Calculate output length
    int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    // Create output tensor
    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Total number of output elements
    int total = N * C_out * L_out;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    conv1d_forward_kernel_gridstride<<<gridSize, blockSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        (int)stride, (int)padding, (int)dilation, (int)groups, (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_gridstride failed: ", cudaGetErrorString(err));

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
            return conv1d_forward_impl(x, weight, bias, stride, padding, dilation, groups);
        },
        "1D Convolution forward pass using grid-stride loops for dynamic workload handling (CUDA)"
    );
}
