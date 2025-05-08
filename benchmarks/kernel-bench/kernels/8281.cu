#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <algorithm>
#include <stdint.h>

namespace py = pybind11;

// Define maximum constant memory size in floats (64KB / 4 bytes = 16384 floats)
#define MAX_CONST_SIZE 16384

// Declare constant memory for storing the weights (read-only data)
__constant__ float const_weight[MAX_CONST_SIZE];

// Kernel: vectorized 1D convolution using constant memory for weights
__global__ void conv1d_forward_kernel_vectorized_const(
    const float* __restrict__ x,
    const float* __restrict__ bias_ptr, // bias pointer, can be null
    float* __restrict__ y,
    const int N,         // batch size
    const int C_in,      // input channels
    const int L_in,      // input length
    const int C_out,     // output channels
    const int K,         // kernel size
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out      // output length
) {
    // Each thread processes 4 output elements at once
    int tid = threadIdx.x;
    int out_ch = blockIdx.x;            // each block in x-dim corresponds to an output channel
    int batch_idx = blockIdx.z;         // each block in z-dim corresponds to a batch element
    int base_out_pos = (blockIdx.y * blockDim.x + tid) * 4;

    if (base_out_pos >= L_out) return;

    // Determine group information
    int group_size_out = C_out / groups;
    int group_size_in  = C_in / groups;
    int group_idx = out_ch / group_size_out;

    // Initialize accumulator for 4 output positions
    float4 output = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float* out_acc = reinterpret_cast<float*>(&output);

    // Iterate over the input channels in the corresponding group
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        // Compute base index in constant memory for this output channel's weights
        int weight_base = out_ch * (group_size_in * K) + local_in_ch * K;

        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float w_val = const_weight[weight_base + k];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int out_pos = base_out_pos + i;
                if (out_pos < L_out) {
                    int in_pos = out_pos * stride + k * dilation - padding;
                    if (in_pos >= 0 && in_pos < L_in) {
                        int x_idx = batch_idx * (C_in * L_in) + in_ch * L_in + in_pos;
                        out_acc[i] += x[x_idx] * w_val;
                    }
                }
            }
        }
    }

    // Add bias if provided
    if (bias_ptr) {
        float b = bias_ptr[out_ch];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            out_acc[i] += b;
        }
    }

    // Write the computed output back to global memory
    int out_idx = batch_idx * (C_out * L_out) + out_ch * L_out + base_out_pos;
    int remaining = min(4, L_out - base_out_pos);
    
    // Use vectorized store if possible and aligned
    if (remaining == 4 && ((uintptr_t)(&y[out_idx]) & 15) == 0) {
        *((float4 *)(&y[out_idx])) = output;
    } else {
        for (int i = 0; i < remaining; ++i) {
            y[out_idx + i] = out_acc[i];
        }
    }
}

// Host implementation
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

    // x shape: [N, C_in, L_in]
    auto x_sizes = x.sizes();
    int N    = x_sizes[0];
    int C_in = x_sizes[1];
    int L_in = x_sizes[2];

    // weight shape: [C_out, C_in/groups, K]
    auto w_sizes = weight.sizes();
    int C_out = w_sizes[0];
    int K     = w_sizes[2];

    // Compute group sizes
    int group_size_in = C_in / groups;
    int total_weight_elems = C_out * group_size_in * K;
    TORCH_CHECK(total_weight_elems <= MAX_CONST_SIZE, "Weight tensor does not fit into constant memory");

    // Calculate output length
    int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options());

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Copy weight tensor to constant memory
    cudaError_t err = cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), total_weight_elems * sizeof(float));
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol failed: ", cudaGetErrorString(err));

    // Launch configuration: vectorized kernel processing 4 output elements per thread
    const int threads_per_block = 128;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(
        C_out, 
        (L_out + (threads_per_block * 4) - 1) / (threads_per_block * 4),
        N
    );

    conv1d_forward_kernel_vectorized_const<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        (int)stride, (int)padding, (int)dilation, (int)groups, L_out
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_vectorized_const failed: ", cudaGetErrorString(err));
    
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
        "Vectorized 1D Convolution forward using constant memory for weights (CUDA)"
    );
}
