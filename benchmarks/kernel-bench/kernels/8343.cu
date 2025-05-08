#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

#define TILE_SIZE 32
#define BLOCK_SIZE 256

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
    __shared__ float s_input[TILE_SIZE][TILE_SIZE + 1];  // +1 for bank conflict avoidance
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Calculate base indices for this block
    const int out_idx_base = bid * BLOCK_SIZE;
    const int n_base = out_idx_base / (C_out * L_out);
    const int tmp = out_idx_base % (C_out * L_out);
    const int c_out_base = tmp / L_out;
    const int l_out_base = tmp % L_out;
    
    // Calculate group parameters
    const int group_size_out = C_out / groups;
    const int group_size_in = C_in / groups;
    const int group_idx = c_out_base / group_size_out;
    
    // Process multiple elements per thread
    #pragma unroll 4
    for (int i = 0; i < BLOCK_SIZE/32; i++) {
        const int local_idx = i * 32 + lane_id;
        if (local_idx + out_idx_base >= N * C_out * L_out) continue;
        
        const int n = n_base + (local_idx / (C_out * L_out));
        const int c_out = c_out_base + ((local_idx % (C_out * L_out)) / L_out);
        const int l_out = l_out_base + (local_idx % L_out);
        
        float sum = 0.0f;
        
        // Process input channels in tiles
        for (int tile = 0; tile < group_size_in; tile += TILE_SIZE) {
            // Collaborative loading of input data into shared memory
            #pragma unroll
            for (int j = 0; j < K; j++) {
                const int in_pos = l_out * stride + j * dilation - padding;
                if (in_pos >= 0 && in_pos < L_in && tid < TILE_SIZE) {
                    const int in_ch = group_idx * group_size_in + tile + tid;
                    if (in_ch < (group_idx + 1) * group_size_in) {
                        s_input[tid][j] = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                    }
                }
            }
            __syncthreads();
            
            // Compute convolution using shared memory
            const int local_in_ch_start = tile;
            const int local_in_ch_end = min(tile + TILE_SIZE, group_size_in);
            
            #pragma unroll
            for (int local_in_ch = local_in_ch_start; local_in_ch < local_in_ch_end; local_in_ch++) {
                #pragma unroll
                for (int k = 0; k < K; k++) {
                    const int in_pos = l_out * stride + k * dilation - padding;
                    if (in_pos >= 0 && in_pos < L_in) {
                        const float w_val = w[c_out * (group_size_in * K) + 
                                           (local_in_ch % group_size_in) * K + k];
                        sum += s_input[local_in_ch - tile][k] * w_val;
                    }
                }
            }
            __syncthreads();
        }
        
        // Add bias if present
        if (bias_ptr != nullptr) {
            sum += bias_ptr[c_out];
        }
        
        // Write result with coalesced access
        y[n * (C_out * L_out) + c_out * L_out + l_out] = sum;
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

    auto y = torch::empty({N, C_out, L_out}, x.options());

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    int total_elements = N * C_out * L_out;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv1d_forward_kernel<<<num_blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", 
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
          "Coalesced and tiled 1D Convolution forward (CUDA)");
}