#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Optimized 1D convolution kernel combining shared memory tiling for weights
// and warp shuffle for efficient weight broadcasting
__global__ void conv1d_forward_kernel_optimized(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr,  // May be nullptr
    float* __restrict__ y,
    const int N,      // Batch size
    const int C_in,   // Input channels
    const int L_in,   // Input length
    const int C_out,  // Output channels
    const int K,      // Kernel size
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out   // Output length
) {
    // Each block processes multiple output channels for one batch element
    // Block grid: blockIdx.x -> batch index, blockIdx.y -> channel tile index
    int n = blockIdx.x;
    int out_ch = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_ch >= C_out) return;

    // Determine channel grouping
    int group_size_out = C_out / groups;
    int group_size_in  = C_in / groups;
    int group_idx = out_ch / group_size_out;

    // The weight tile for this output channel has dimensions: [group_size_in, K]
    int tile_size = group_size_in * K;

    // Allocate shared memory for weight tiles with padding to avoid bank conflicts
    extern __shared__ float shw[]; // Size: blockDim.y * (tile_size + 1) to add padding
    // Each thread (with fixed y) cooperatively loads the weight tile for its output channel
    // Add padding after each row to avoid bank conflicts
    const int padded_tile_size = tile_size + 1;  // Add 1 float padding per row
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        shw[threadIdx.y * padded_tile_size + i] = w[out_ch * tile_size + i];
    }
    __syncthreads();

    // Use warp-level primitives to distribute output positions among threads
    const unsigned warp_size = 32;
    int lane_id = threadIdx.x % warp_size;        // lane in current warp
    int warp_id = threadIdx.x / warp_size;          // warp index within the x-dimension

    // Determine how many warps participate per output channel row
    int total_warps_per_row = blockDim.x / warp_size;
    
    // Each warp processes a contiguous tile of output positions
    for (int base = warp_id * warp_size; base < L_out; base += total_warps_per_row * warp_size) {
        int out_pos = base + lane_id;
        if (out_pos < L_out) {
            float sum = 0.0f;
            // Loop over the input channels for this group
            for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
                int in_ch = group_idx * group_size_in + local_in_ch;
                for (int k = 0; k < K; k++) {
                    int in_pos = out_pos * stride + k * dilation - padding;
                    float x_val = 0.0f;
                    if (in_pos >= 0 && in_pos < L_in) {
                        x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                    }
                    // Use warp shuffle to broadcast the weight value from lane 0 to all lanes
                    float w_val;
                    int weight_idx = local_in_ch * K + k;
                    if (lane_id == 0) {
                        w_val = shw[threadIdx.y * tile_size + weight_idx];
                    }
                    w_val = __shfl_sync(0xffffffff, w_val, 0);
                    
                    sum += x_val * w_val;
                }
            }
            // Optionally add bias
            if (bias_ptr) {
                sum += bias_ptr[out_ch];
            }
            y[n * (C_out * L_out) + out_ch * L_out + out_pos] = sum;
        }
    }
}

// Host function for launching the optimized kernel
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

    // Input dimensions: [N, C_in, L_in]
    auto x_sizes = x.sizes();
    int64_t N    = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    // Weight dimensions: [C_out, C_in/groups, K]
    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K     = w_sizes[2];

    // Calculate the output length
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Define block and grid dimensions
    // We choose a 2D block: x-dimension covers output position tiling using warps, y-dimension covers multiple output channels
    const int warp_size = 32;
    const int WARPS_PER_BLOCK = 4;      // Number of warps along x-dim
    const int CHANNELS_PER_BLOCK = 4;     // Number of output channels per block
    dim3 block(warp_size * WARPS_PER_BLOCK, CHANNELS_PER_BLOCK);
    dim3 grid(N, (C_out + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK);

    // Shared memory: one weight tile per output channel in the block
    int group_size_in = C_in / groups;
    int tile_size = group_size_in * K;
    int sharedMem = CHANNELS_PER_BLOCK * tile_size * sizeof(float);

    conv1d_forward_kernel_optimized<<<grid, block, sharedMem>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        (int)N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups, (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_optimized failed: ", cudaGetErrorString(err));

    return y;
}

// Pybind11 binding
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
              return conv1d_forward_impl_optimized(x, weight, bias, stride, padding, dilation, groups);
          },
          "Optimized 1D Convolution forward (CUDA) combining tiling and warp-shuffle optimizations");
}
