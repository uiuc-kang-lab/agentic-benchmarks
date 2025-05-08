#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Tiling parameters
#define TILE_WIDTH 32
#define TILE_OC    8

// Kernel that caches group-specific weights in shared memory. 
// Each block processes a tile of output positions (TILE_WIDTH) and a tile of output channels (TILE_OC) for a given batch element and group.

// Expected grid dimensions:
//   grid.x = (L_out + TILE_WIDTH - 1) / TILE_WIDTH
//   grid.y = (group_size_out + TILE_OC - 1) / TILE_OC
//   grid.z = N * groups   (with n = blockIdx.z / groups, g = blockIdx.z % groups)

// Shared memory size: group_size_out * group_size_in * K * sizeof(float)

__global__ void conv1d_forward_kernel_tiled(
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
    const int L_out,
    const int group_size_in,
    const int group_size_out
) {
    // Allocate shared memory for weights of the current group
    extern __shared__ float w_shared[];  // size = group_size_out * group_size_in * K

    // Determine batch and group from grid.z
    int block_idx_z = blockIdx.z;
    int n = block_idx_z / groups;    // batch index
    int g = block_idx_z % groups;      // current group index

    // Determine tile base for output positions and output channels within the group
    int out_pos_base = blockIdx.x * TILE_WIDTH;
    int out_ch_base = blockIdx.y * TILE_OC;

    // Thread indices within the block tile
    int t_x = threadIdx.x; // for output positions
    int t_y = threadIdx.y; // for output channels (within group)

    // Compute global output indices (within L_out and within group)
    int out_pos = out_pos_base + t_x;              // output position index
    int out_ch_local = out_ch_base + t_y;            // local output channel index within group

    // Load the group-specific weights into shared memory.
    // Global weight layout: [C_out, group_size_in, K] where C_out = groups * group_size_out.
    // For group g, the weights start at offset: g * (group_size_out * group_size_in * K)
    int total_weights = group_size_out * group_size_in * K;
    int thread_id = t_y * TILE_WIDTH + t_x;  // linear thread id in block (blockDim.x = TILE_WIDTH, blockDim.y = TILE_OC)
    
    for (int i = thread_id; i < total_weights; i += (TILE_WIDTH * TILE_OC)) {
        w_shared[i] = w[ g * (group_size_out * group_size_in * K) + i ];
    }
    __syncthreads(); // Synchronize to ensure all weights are loaded

    // Only compute if within output boundaries
    if (out_pos < L_out && out_ch_local < group_size_out) {
        int global_out_ch = g * group_size_out + out_ch_local; // Global output channel index
        float sum = 0.0f;
        
        // Iterate over input channels for the current group
        for (int lc = 0; lc < group_size_in; ++lc) {
            int in_ch = g * group_size_in + lc; // Global input channel
            
            // For each kernel element, compute the convolution sum
            for (int k = 0; k < K; ++k) {
                int in_pos = out_pos * stride + k * dilation - padding;
                if (in_pos >= 0 && in_pos < L_in) {
                    float x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                    // Calculate index into shared memory for weights
                    int w_index = (out_ch_local * group_size_in + lc) * K + k;
                    float w_val = w_shared[w_index];
                    sum += x_val * w_val;
                }
            }
        }
        
        // Add bias if provided
        if (bias_ptr) {
            sum += bias_ptr[global_out_ch];
        }
        
        // Write the result to the output tensor
        y[n * (C_out * L_out) + global_out_ch * L_out + out_pos] = sum;
    }
}

// Host implementation of the conv1d forward operation
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
    TORCH_CHECK(x.scalar_type() == at::kFloat(), "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat(), "weight must be float32");

    int64_t N = x.size(0);
    int64_t C_in = x.size(1);
    int64_t L_in = x.size(2);
    int64_t C_out = weight.size(0);
    int64_t K = weight.size(2);

    // Compute output length for 1D convolution
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated L_out is non-positive");

    auto y = torch::empty({N, C_out, L_out}, x.options());

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat(), "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    int group_size_in = C_in / groups;
    int group_size_out = C_out / groups;

    // Set up grid dimensions:
    // grid.x: tile over output positions
    // grid.y: tile over output channels within a group
    // grid.z: each block is assigned to a (batch, group) pair
    dim3 block(TILE_WIDTH, TILE_OC);
    dim3 grid;
    grid.x = (L_out + TILE_WIDTH - 1) / TILE_WIDTH;
    grid.y = (group_size_out + TILE_OC - 1) / TILE_OC;
    grid.z = N * groups;

    // Shared memory size (in bytes) for the weights of one group
    size_t shared_mem_bytes = group_size_out * group_size_in * K * sizeof(float);

    conv1d_forward_kernel_tiled<<<grid, block, shared_mem_bytes>>>(
         x.data_ptr<float>(),
         weight.data_ptr<float>(),
         bias_ptr,
         y.data_ptr<float>(),
         N, C_in, L_in, C_out, K,
         (int)stride, (int)padding, (int)dilation, (int)groups, (int)L_out,
         group_size_in, group_size_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

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
        "Tiled 1D Convolution forward (CUDA) with shared memory cached weights"
    );
}
