#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// This kernel experiments with variable block sizes (e.g., 32, 64, 128, 256, 512) to identify the optimal configuration.
// Each block handles one output channel and a subset of output positions. The kernel loads the weight kernel into shared memory to reduce global memory accesses.

__global__ void conv1d_forward_kernel_experimental(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr, // optional bias
    float* __restrict__ y,
    const int N,       // batch size
    const int C_in,    // number of input channels
    const int L_in,    // input length
    const int C_out,   // number of output channels
    const int K,       // kernel size
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out    // output length
) {
    // Map each block to one output channel and a subset of output positions
    int out_ch = blockIdx.x;
    int out_pos = blockIdx.y * blockDim.x + threadIdx.x;
    int n = blockIdx.z;
    if (out_pos >= L_out) return;

    int group_size_out = C_out / groups;
    int group_size_in = C_in / groups;
    int group_idx = out_ch / group_size_out;

    // Use shared memory to load the weight kernel for the current output channel
    extern __shared__ float shmem[];  // shared memory size: group_size_in * K floats
    int total_weights = group_size_in * K;
    for (int i = threadIdx.x; i < total_weights; i += blockDim.x) {
        shmem[i] = w[out_ch * total_weights + i];
    }
    __syncthreads();

    float sum = 0.0f;
    // For each input channel in the group and for each kernel position, accumulate the results
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        for (int k = 0; k < K; ++k) {
            int in_pos = out_pos * stride + k * dilation - padding;
            if (in_pos >= 0 && in_pos < L_in) {
                float x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                float w_val = shmem[local_in_ch * K + k];
                sum += x_val * w_val;
            }
        }
    }

    if (bias_ptr) {
        sum += bias_ptr[out_ch];
    }
    y[n * (C_out * L_out) + out_ch * L_out + out_pos] = sum;
}

// Host function that sets up the grid and block dimensions with an experimental block size
at::Tensor conv1d_forward_impl_experimental(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    int block_size  // experimental block size (e.g., 32, 64, 128, 256, 512)
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    auto x_sizes = x.sizes();
    int64_t N = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K = w_sizes[2];

    // Compute the output length
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Set up grid and block dimensions based on the experimental block size
    dim3 blockDim(block_size);
    dim3 gridDim(C_out, (L_out + block_size - 1) / block_size, N);

    int group_size_in = C_in / groups;
    size_t sharedMemSize = group_size_in * K * sizeof(float);

    conv1d_forward_kernel_experimental<<<gridDim, blockDim, sharedMemSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups, (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_experimental failed: ", cudaGetErrorString(err));

    return y;
}

// Pybind11 module registration. The "forward" function now accepts an extra block_size parameter (defaulting to 256) for experiments.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](at::Tensor x,
           at::Tensor weight,
           py::object bias_obj,
           int64_t stride,
           int64_t padding,
           int64_t dilation,
           int64_t groups,
           int block_size) {
            c10::optional<at::Tensor> bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<at::Tensor>();
            }
            return conv1d_forward_impl_experimental(x, weight, bias, stride, padding, dilation, groups, block_size);
        },
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias_obj"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"),
        py::arg("block_size") = 256,
        "Forward pass of 1D convolution with experimental block size tuning"
    );
}
