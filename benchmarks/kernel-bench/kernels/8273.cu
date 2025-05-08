#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Constants for warp-level processing
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int POSITIONS_PER_THREAD = 4;

__global__ void conv1d_forward_kernel_balanced(
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
    // Calculate warp and lane IDs
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Each block handles multiple output channels and positions
    const int out_ch_base = blockIdx.x * WARPS_PER_BLOCK;
    const int out_ch = out_ch_base + warp_id;
    const int batch_idx = blockIdx.z;

    // Skip if this warp's output channel is out of bounds
    if (out_ch >= C_out) return;

    // Calculate group information
    const int group_size_out = C_out / groups;
    const int group_size_in = C_in / groups;
    const int group_idx = out_ch / group_size_out;
    const int in_ch_start = group_idx * group_size_in;

    // Each thread processes multiple output positions
    const int positions_per_block = WARP_SIZE * POSITIONS_PER_THREAD;
    const int block_start_pos = blockIdx.y * positions_per_block;
    
    // Register array to accumulate results
    float accum[POSITIONS_PER_THREAD] = {0.0f};
    
    // Pre-load bias if present
    const float bias_val = bias_ptr ? bias_ptr[out_ch] : 0.0f;

    // Process input channels in chunks to maximize register reuse
    for (int in_ch_offset = 0; in_ch_offset < group_size_in; ++in_ch_offset) {
        const int in_ch = in_ch_start + in_ch_offset;
        
        // Cache weights in registers
        float weight_cache[32];  // Assuming K <= 32
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            weight_cache[k] = w[out_ch * (group_size_in * K) + in_ch_offset * K + k];
        }

        // Process output positions assigned to this thread
        #pragma unroll
        for (int p = 0; p < POSITIONS_PER_THREAD; ++p) {
            const int out_pos = block_start_pos + lane_id + p * WARP_SIZE;
            if (out_pos < L_out) {
                // Compute convolution for this position
                #pragma unroll
                for (int k = 0; k < K; ++k) {
                    const int in_pos = out_pos * stride + k * dilation - padding;
                    if (in_pos >= 0 && in_pos < L_in) {
                        const float x_val = x[batch_idx * (C_in * L_in) + in_ch * L_in + in_pos];
                        accum[p] += x_val * weight_cache[k];
                    }
                }
            }
        }
    }

    // Write results to global memory
    #pragma unroll
    for (int p = 0; p < POSITIONS_PER_THREAD; ++p) {
        const int out_pos = block_start_pos + lane_id + p * WARP_SIZE;
        if (out_pos < L_out) {
            const int out_idx = batch_idx * (C_out * L_out) + out_ch * L_out + out_pos;
            y[out_idx] = accum[p] + bias_val;
        }
    }
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
    int64_t N = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K = w_sizes[2];

    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Configure grid dimensions for balanced workload
    const int positions_per_block = WARP_SIZE * POSITIONS_PER_THREAD;
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (C_out + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
        (L_out + positions_per_block - 1) / positions_per_block,
        N
    );

    conv1d_forward_kernel_balanced<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_balanced failed: ", cudaGetErrorString(err));

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
        "Balanced workload 1D Convolution forward (CUDA)"
    );
}