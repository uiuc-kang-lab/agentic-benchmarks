#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define tile dimensions for spatial tiling
#define TILE_W 8
#define TILE_H 8

// Kernel: Each block computes a tile of the output for a given (n, c_out) pair.
// The kernel caches the weight for the given output channel (and associated input channels) into shared memory.
// Only one __syncthreads() is used after loading shared memory to ensure consistency.

__global__ void conv2d_cuda_kernel_tiled_shared_weight(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    // Determine the batch (n) and output channel (c_out) for this block
    int n = blockIdx.x;       // Batch index
    int c_out = blockIdx.y;   // Output channel index

    // Determine tile index for the spatial output using blockIdx.z
    int tiles_w = (W_out + TILE_W - 1) / TILE_W; // number of tiles in width
    int tile_idx = blockIdx.z;
    int tile_row = tile_idx / tiles_w;
    int tile_col = tile_idx % tiles_w;

    // Starting output coordinates for this tile
    int h_out_start = tile_row * TILE_H;
    int w_out_start = tile_col * TILE_W;

    // Thread indices within the block (2D block layout: TILE_W x TILE_H)
    int local_x = threadIdx.x; // column index within tile
    int local_y = threadIdx.y; // row index within tile
    int w_out = w_out_start + local_x;
    int h_out = h_out_start + local_y;

    // Load the weight for this output channel and its corresponding input channels into shared memory.
    // Determine the group for c_out and the corresponding input channel range.
    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int C_in_per_group = C_in / groups;  // Number of input channels in this group
    int weight_elements = C_in_per_group * K_h * K_w;  // Total weight elements for this output channel

    extern __shared__ float s_weight[];  // Shared memory for weight caching

    // Use linear thread indexing within the block to load weight values
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;
    for (int i = tid; i < weight_elements; i += block_threads) {
        int local_c = i / (K_h * K_w);  // index within the input channel subset
        int rem = i % (K_h * K_w);
        int kh = rem / K_w;
        int kw = rem % K_w;
        // Global weight layout: [C_out, C_in_per_group, K_h, K_w]
        int global_index = ((c_out * C_in_per_group + local_c) * K_h + kh) * K_w + kw;
        s_weight[i] = weight[global_index];
    }
    // Synchronize threads to ensure the weight is fully loaded
    __syncthreads();

    // Only compute if the output coordinate is within bounds
    if (h_out < H_out && w_out < W_out) {
        float out_val = (bias != nullptr) ? bias[c_out] : 0.0f;
        // Iterate over the input channels of the current group
        for (int local_c = 0; local_c < C_in_per_group; ++local_c) {
            int c_in = c_in_start + local_c;
            // Compute the base offset for this input channel
            int input_base = ((n * C_in + c_in) * H_in * W_in);
            // Iterate over the kernel height and width
            for (int kh = 0; kh < K_h; ++kh) {
                int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                if (h_in < 0 || h_in >= H_in) continue;
                int input_row = input_base + h_in * W_in;
                for (int kw = 0; kw < K_w; ++kw) {
                    int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                    if (w_in < 0 || w_in >= W_in) continue;
                    int weight_idx = ((local_c * K_h) + kh) * K_w + kw;
                    out_val += input[input_row + w_in] * s_weight[weight_idx];
                }
            }
        }
        // Write the result to the output tensor
        int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
        output[output_idx] = out_val;
    }
}

// C++ interface exposed via Pybind11
torch::Tensor conv2d_cuda_tiled_shared_weight(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // Ensure tensors are contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    if (bias_opt.has_value()) {
        bias_opt.value() = bias_opt.value().contiguous();
    }

    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "Bias tensor must be on CUDA if provided");
    }

    // Input dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    // Weight dimensions (assumed layout: [C_out, C_in/groups, K_h, K_w])
    int C_out = weight.size(0);
    int K_h = weight.size(2);
    int K_w = weight.size(3);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int padding_h = padding[0];
    int padding_w = padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    // Compute output spatial dimensions
    int H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Grid dimensions:
    //   grid.x: batch size (N)
    //   grid.y: output channels (C_out)
    //   grid.z: number of spatial tiles = ceil(H_out / TILE_H) * ceil(W_out / TILE_W)
    int tiles_y = (H_out + TILE_H - 1) / TILE_H;
    int tiles_x = (W_out + TILE_W - 1) / TILE_W;
    int grid_z = tiles_y * tiles_x;

    dim3 grid(N, C_out, grid_z);
    dim3 block(TILE_W, TILE_H);

    // Shared memory size: weight cache for one output channel (for its input group):
    int C_in_per_group = C_in / groups;
    int weight_elements = C_in_per_group * K_h * K_w;
    size_t shared_mem_size = weight_elements * sizeof(float);

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        bias_ptr = bias_opt.value().data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    conv2d_cuda_kernel_tiled_shared_weight<<<grid, block, shared_mem_size>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in conv2d_cuda_kernel_tiled_shared_weight: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_tiled_shared_weight, "Tiled 2D convolution with shared weight caching (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}
