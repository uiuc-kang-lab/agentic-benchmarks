#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_BLOCK 32

__global__ void conv1d_forward_kernel(
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
    __shared__ float s_input[ELEMENTS_PER_BLOCK * WARP_SIZE];
    __shared__ float s_weight[32 * 32];  // Cache for weights

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Calculate base output position for this block
    const int block_start = bid * ELEMENTS_PER_BLOCK;
    const int out_pos_base = block_start % L_out;
    const int out_ch = (block_start / L_out) % C_out;
    const int n = block_start / (L_out * C_out);

    if (n >= N) return;

    // Determine group information
    const int group_size_out = C_out / groups;
    const int group_size_in = C_in / groups;
    const int group_idx = out_ch / group_size_out;

    float results[ELEMENTS_PER_BLOCK] = {0.0f};

    // Process input channels in chunks
    for (int in_ch_base = 0; in_ch_base < group_size_in; in_ch_base += WARP_SIZE) {
        // Load weights into shared memory
        if (lane_id + in_ch_base < group_size_in && tid < K) {
            s_weight[tid * WARP_SIZE + lane_id] = 
                w[out_ch * (group_size_in * K) + (lane_id + in_ch_base) * K + tid];
        }
        __syncthreads();

        // Process ELEMENTS_PER_BLOCK elements per thread block
        for (int elem = 0; elem < ELEMENTS_PER_BLOCK; elem++) {
            float val = 0.0f;
            const int out_pos = out_pos_base + elem;
            if (out_pos >= L_out) continue;

            // Load input data into shared memory
            for (int k = tid; k < K; k += BLOCK_SIZE) {
                const int in_pos = out_pos * stride + k * dilation - padding;
                if (in_pos >= 0 && in_pos < L_in) {
                    for (int ic = 0; ic < WARP_SIZE && (ic + in_ch_base) < group_size_in; ic++) {
                        const int in_ch = group_idx * group_size_in + in_ch_base + ic;
                        s_input[k * WARP_SIZE + ic] = 
                            x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                    }
                }
            }
            __syncthreads();

            // Compute convolution using shared memory
            for (int k = 0; k < K; k++) {
                const int in_pos = out_pos * stride + k * dilation - padding;
                if (in_pos >= 0 && in_pos < L_in) {
                    for (int ic = 0; ic < WARP_SIZE && (ic + in_ch_base) < group_size_in; ic++) {
                        val += s_input[k * WARP_SIZE + ic] * 
                               s_weight[k * WARP_SIZE + ic];
                    }
                }
            }

            // Warp-level reduction
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }

            if (lane_id == 0) {
                results[elem] += val;
            }
            __syncthreads();
        }
    }

    // Write results to global memory
    if (lane_id == 0) {
        for (int elem = 0; elem < ELEMENTS_PER_BLOCK; elem++) {
            const int out_pos = out_pos_base + elem;
            if (out_pos < L_out) {
                float final_val = results[elem];
                if (bias_ptr) {
                    final_val += bias_ptr[out_ch];
                }
                y[n * (C_out * L_out) + out_ch * L_out + out_pos] = final_val;
            }
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

    int total_elements = N * C_out * L_out;
    int num_blocks = (total_elements + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    conv1d_forward_kernel<<<num_blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel failed: ", cudaGetErrorString(err));

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
        "Optimized 1D Convolution forward (CUDA) with shared memory and warp primitives"
    );
}