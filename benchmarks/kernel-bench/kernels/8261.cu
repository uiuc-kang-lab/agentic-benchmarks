#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Optimized 1D convolution kernel using improved thread mapping and shared memory for weights
__global__ void conv1d_forward_kernel_optimized(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr, // optional bias
    float* __restrict__ y,
    const int N,        // batch size
    const int C_in,     // number of input channels
    const int L_in,     // input length
    const int C_out,    // number of output channels
    const int K,        // kernel size
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out     // output length
) {
    // Map each block: one output channel and a subset of positions
    int out_ch = blockIdx.x;
    int out_pos = blockIdx.y * blockDim.x + threadIdx.x;
    int n = blockIdx.z;
    if (out_pos >= L_out) return;

    // Determine group information
    int group_size_out = C_out / groups;
    int group_size_in  = C_in / groups;
    int group_idx      = out_ch / group_size_out;

    // Use shared memory to load the weight kernel for this output channel once per block
    // Each output channel has a weight kernel of size (group_size_in * K)
    extern __shared__ float shmem[];  // size: group_size_in * K floats
    int total_weights = group_size_in * K;
    // Each thread loads multiple elements if needed
    for (int i = threadIdx.x; i < total_weights; i += blockDim.x) {
        shmem[i] = w[out_ch * total_weights + i];
    }
    __syncthreads();

    float sum = 0.0f;
    // Loop over the input channels in the corresponding group and over kernel positions
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        for (int k = 0; k < K; ++k) {
            int in_pos = out_pos * stride + k * dilation - padding;
            if (in_pos >= 0 && in_pos < L_in) {
                float x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                // Load weight from shared memory; each output channel's kernel is stored contiguously
                float w_val = shmem[local_in_ch * K + k];
                sum += x_val * w_val;
            }
        }
    }

    // Add bias if provided
    if (bias_ptr) {
        sum += bias_ptr[out_ch];
    }

    // Write the result to output tensor
    y[n * (C_out * L_out) + out_ch * L_out + out_pos] = sum;
}

// Host implementation that sets up tensor dimensions, grid, block, and shared memory size
at::Tensor conv1d_forward_impl_optimized(
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

    // Get dimensions
    auto x_sizes = x.sizes();
    int64_t N    = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K     = w_sizes[2];

    // Compute output length
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Launch configuration: use a 3D grid with dimensions: output channel, output position, batch.
    dim3 blockSize(256);
    dim3 gridSize(C_out, (L_out + blockSize.x - 1) / blockSize.x, N);

    // Calculate shared memory size: each block loads weights for one output channel
    int group_size_in = C_in / groups;
    size_t sharedMemSize = group_size_in * K * sizeof(float);

    conv1d_forward_kernel_optimized<<<gridSize, blockSize, sharedMemSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_optimized failed: ", cudaGetErrorString(err));
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
            return conv1d_forward_impl_optimized(x, weight, bias, stride, padding, dilation, groups);
        },
        "Optimized 1D Convolution forward (CUDA) using improved thread mapping and shared memory for weights"
    );
}
