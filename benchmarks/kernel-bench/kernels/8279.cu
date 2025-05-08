#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <stdint.h>
#include <algorithm>

namespace py = pybind11;

// Helper: Vectorized store of float4
__device__ __forceinline__ void store_float4(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr) = val;
}

// Combined kernel: uses shared memory to cache weights and vectorized processing for output positions
__global__ void conv1d_forward_kernel_combined(
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
    // Each block is responsible for one output channel and a set of output positions (vectorized by 4)
    const int out_ch = blockIdx.x;
    const int batch = blockIdx.z;

    // Determine group information
    const int group_size_out = C_out / groups;
    const int group_size_in  = C_in / groups;
    const int group_idx      = out_ch / group_size_out;

    // Allocate shared memory for current output channel's weights
    // Size = group_size_in * K
    extern __shared__ float shmem[];
    const int total_weights = group_size_in * K;
    for (int i = threadIdx.x; i < total_weights; i += blockDim.x) {
        shmem[i] = w[out_ch * total_weights + i];
    }
    __syncthreads();

    // Each thread handles 4 output positions
    int thread_base = blockIdx.y * blockDim.x + threadIdx.x;  // index in terms of 4-element groups
    int base_out = thread_base * 4;
    if (base_out >= L_out) return;

    // Initialize accumulator as a float4 vector
    float4 output;
    output.x = 0.0f; output.y = 0.0f; output.z = 0.0f; output.w = 0.0f;
    float* acc_ptr = reinterpret_cast<float*>(&output);

    // Loop over input channels within the current group
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        // Loop over kernel positions
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float w_val = shmem[local_in_ch * K + k];
            // Process 4 vectorized output positions
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int out_pos = base_out + i;
                if (out_pos < L_out) {
                    int in_pos = out_pos * stride + k * dilation - padding;
                    if (in_pos >= 0 && in_pos < L_in) {
                        int x_index = batch * (C_in * L_in) + in_ch * L_in + in_pos;
                        acc_ptr[i] += x[x_index] * w_val;
                    }
                }
            }
        }
    }

    // Add bias if available
    if (bias_ptr) {
        float b = bias_ptr[out_ch];
        acc_ptr[0] += b;
        acc_ptr[1] += b;
        acc_ptr[2] += b;
        acc_ptr[3] += b;
    }

    // Write back results into y
    int out_index = batch * (C_out * L_out) + out_ch * L_out + base_out;
    int remaining = L_out - base_out;
    if (remaining >= 4 && ((uintptr_t)(&y[out_index]) & 15) == 0) {
        store_float4(&y[out_index], output);
    } else {
        for (int i = 0; i < std::min(remaining, 4); ++i) {
            y[out_index + i] = acc_ptr[i];
        }
    }
}

// Host function: sets up grid/block dimensions and launches the combined kernel
at::Tensor conv1d_forward_impl_combined(
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

    // Get input dimensions
    auto x_sizes = x.sizes();
    int N    = x_sizes[0];
    int C_in = x_sizes[1];
    int L_in = x_sizes[2];

    auto w_sizes = weight.sizes();
    int C_out = w_sizes[0];
    int K     = w_sizes[2];

    // Compute output length
    int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Configure launch parameters
    const int threads_per_block = 128;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(
        C_out,  // one block per output channel
        (L_out + threads_per_block * 4 - 1) / (threads_per_block * 4),  // blocks covering all output positions
        N  // one block row per batch element
    );

    int group_size_in = C_in / groups;
    size_t sharedMemSize = group_size_in * K * sizeof(float);

    // Launch the combined kernel
    conv1d_forward_kernel_combined<<<grid_dim, block_dim, sharedMemSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        (int)stride, (int)padding, (int)dilation, (int)groups, L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_combined failed: ", cudaGetErrorString(err));
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
            return conv1d_forward_impl_combined(x, weight, bias, stride, padding, dilation, groups);
        },
        "Combined optimized and vectorized 1D Convolution forward (CUDA)"
    );
}
