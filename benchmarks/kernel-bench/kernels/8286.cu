#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <stdint.h>

namespace py = pybind11;

// Define vectorization factor
#define VEC 4

// This kernel partitions the reduction over input channels so that multiple thread blocks can cooperatively
// accumulate partial sums for a given output element. Each block computes a partial sum over a slice of input
// channels and then uses atomicAdd (only once per block per output element) to write its result into global memory.
// For blocks processing the first partition (part == 0), the block also adds the bias so that it is incorporated only once.

__global__ void conv1d_forward_kernel_atomic_minimal(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr, // may be nullptr
    float* __restrict__ y,
    const int N,       // batch size
    const int C_in,    // input channels
    const int L_in,    // input length
    const int C_out,   // output channels
    const int K,       // kernel size
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out    // output length
) {
    // Compute group parameters
    int group_size_in = C_in / groups;
    int group_size_out = C_out / groups;
    // Number of output position groups (each group processes VEC output positions)
    int out_pos_groups = (L_out + VEC - 1) / VEC;

    // Decode grid indices
    int out_ch = blockIdx.x;               // each block in x-dim corresponds to one output channel
    int part = blockIdx.y;                 // partition index for input channels
    int global_outgrp = blockIdx.z;          // encodes both batch and output position group
    int batch = global_outgrp / out_pos_groups;
    int out_grp = global_outgrp % out_pos_groups;
    int base_out = out_grp * VEC;            // starting output position for this block

    // Each block processes a subset of the input channels for the given (batch, out_ch, output position group)
    int tid = threadIdx.x;
    int local_idx = part * blockDim.x + tid;    // index within the group input channels

    // Each thread computes partial sums for VEC consecutive output elements
    float partial[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
        partial[i] = 0.0f;
    }

    // Only if the thread corresponds to a valid input channel
    if (local_idx < group_size_in) {
        // Determine the actual input channel index
        int group_idx = out_ch / group_size_out;  // which group this output channel belongs to
        int in_ch = group_idx * group_size_in + local_idx;
        
        // Loop over kernel positions
        for (int k = 0; k < K; k++) {
            // Weight layout: each output channel has a contiguous block of weights (group_size_in * K)
            float w_val = w[out_ch * (group_size_in * K) + local_idx * K + k];
            
            #pragma unroll
            for (int i = 0; i < VEC; i++) {
                int out_pos = base_out + i;
                if (out_pos < L_out) {
                    // Compute corresponding input position
                    int in_pos = out_pos * stride + k * dilation - padding;
                    if (in_pos >= 0 && in_pos < L_in) {
                        int x_index = batch * (C_in * L_in) + in_ch * L_in + in_pos;
                        partial[i] += x[x_index] * w_val;
                    }
                }
            }
        }
    }

    // Use shared memory for block-level reduction. Allocate blockDim.x * VEC floats.
    extern __shared__ float sdata[]; // size: blockDim.x * VEC
    for (int i = 0; i < VEC; i++) {
        sdata[tid * VEC + i] = partial[i];
    }
    __syncthreads();

    // Tree-based reduction across the block's threads
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            for (int i = 0; i < VEC; i++) {
                sdata[tid * VEC + i] += sdata[(tid + s) * VEC + i];
            }
        }
        __syncthreads();
    }

    // Thread 0 in the block writes the block's partial sum to global memory atomically
    if (tid == 0) {
        for (int i = 0; i < VEC; i++) {
            int out_pos = base_out + i;
            if (out_pos < L_out) {
                int index = batch * (C_out * L_out) + out_ch * L_out + out_pos;
                // Add bias only once (from the first partition block)
                float bias_val = 0.0f;
                if (bias_ptr && part == 0) {
                    bias_val = bias_ptr[out_ch];
                }
                atomicAdd(&y[index], sdata[i] + bias_val);
            }
        }
    }
}

// Host wrapper function for the atomic minimal kernel
at::Tensor conv1d_forward_impl_atomic(
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
    int N = x_sizes[0];
    int C_in = x_sizes[1];
    int L_in = x_sizes[2];

    auto w_sizes = weight.sizes();
    int C_out = w_sizes[0];
    int K = w_sizes[2];

    int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    // Allocate output tensor and initialize it to 0
    auto y = torch::zeros({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Grid configuration:
    // - grid.x: one block per output channel (C_out)
    // - grid.y: partitions for the input channels within a group
    int group_size_in = C_in / groups;
    int threads_per_block = 128;
    int num_parts = (group_size_in + threads_per_block - 1) / threads_per_block;
    // - grid.z: one block per (batch, output position group).
    int out_pos_groups = (L_out + VEC - 1) / VEC;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(C_out, num_parts, N * out_pos_groups);

    // Shared memory size per block: threads_per_block * VEC * sizeof(float)
    size_t sharedMemSize = threads_per_block * VEC * sizeof(float);

    conv1d_forward_kernel_atomic_minimal<<<grid_dim, block_dim, sharedMemSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        (int)stride, (int)padding, (int)dilation, (int)groups, L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_atomic_minimal failed: ", cudaGetErrorString(err));

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
            return conv1d_forward_impl_atomic(x, weight, bias, stride, padding, dilation, groups);
        },
        "1D Convolution forward (CUDA) with minimal atomic operations for partial sum accumulation"
    );
}
