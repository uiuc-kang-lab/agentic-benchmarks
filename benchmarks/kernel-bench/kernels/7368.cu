#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define tile dimensions for output spatial tiling
#define TILE_W 16
#define TILE_H 16

// Optimized CUDA kernel that loads the weight filter into shared memory
// Each block is assigned a specific batch element and output channel,
// and computes a TILE_H x TILE_W sub-tile of the output feature map.
// The weight filter for that output channel is loaded into shared memory once
// using a minimal __syncthreads() call for consistency.

__global__ void conv2d_cuda_kernel_shared_weight(
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
    // Each block is responsible for one batch element (n) and one output channel (c_out).
    int n = blockIdx.x;
    int c_out = blockIdx.y;

    // Compute tiling for the output spatial dimensions.
    // The grid's z-dimension indexes the spatial tile.
    int tile_count_x = (W_out + TILE_W - 1) / TILE_W;
    int tile_index = blockIdx.z;
    int tile_y = tile_index / tile_count_x;
    int tile_x = tile_index % tile_count_x;

    // Each thread computes one output pixel within the tile.
    int w_out = tile_x * TILE_W + threadIdx.x;
    int h_out = tile_y * TILE_H + threadIdx.y;
    if (w_out >= W_out || h_out >= H_out) return;

    // Use shared memory to load the weight filter for the current output channel.
    // The filter has dimensions: (C_in/groups, K_h, K_w).
    extern __shared__ float s_weight[];
    int C_in_per_group = C_in / groups;
    // Determine group index based on output channel
    int group = c_out / (C_out / groups);
    int weight_size = C_in_per_group * K_h * K_w;

    // Use all threads in the block to cooperatively load the filter into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    for (int idx = tid; idx < weight_size; idx += num_threads) {
        int c = idx / (K_h * K_w);           // index within the input channels for this group
        int rem = idx % (K_h * K_w);
        int k_h = rem / K_w;
        int k_w = rem % K_w;
        // Global weight index: weights are stored as [C_out, C_in/g, K_h, K_w]
        int weight_idx = (((c_out * C_in + (group * C_in_per_group + c)) * K_h + k_h) * K_w) + k_w;
        s_weight[idx] = weight[weight_idx];
    }
    __syncthreads(); // Ensure that the entire filter is loaded before use

    // Initialize the output value with bias (if provided)
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;
    int c_in_start = group * C_in_per_group;
    
    // Convolution: iterate over the input channels in the group and the kernel window
    for (int c = 0; c < C_in_per_group; ++c) {
        for (int k_h = 0; k_h < K_h; ++k_h) {
            for (int k_w = 0; k_w < K_w; ++k_w) {
                int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
                int w_in = w_out * stride_w - padding_w + k_w * dilation_w;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = ((n * C_in + (c_in_start + c)) * H_in + h_in) * W_in + w_in;
                    // Index into the shared weight array
                    int s_w_idx = ((c * K_h + k_h) * K_w + k_w);
                    value += input[input_idx] * s_weight[s_w_idx];
                }
            }
        }
    }

    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_idx] = value;
}

// C++ interface to the PyTorch module
// This function sets up the grid and block dimensions to make sure that each block
// processes one batch element and one output channel for a tile of the spatial domain.

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // Ensure inputs are contiguous and on CUDA
    input = input.contiguous();
    weight = weight.contiguous();
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "Bias tensor must be on CUDA if provided");
    }

    // Input dimensions: [N, C_in, H_in, W_in]
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    // Weight dimensions: [C_out, C_in/groups, K_h, K_w]
    int64_t C_out = weight.size(0);
    int64_t K_h = weight.size(2);
    int64_t K_w = weight.size(3);

    // Stride, padding, and dilation
    int64_t stride_h = stride[0];
    int64_t stride_w = stride[1];
    int64_t padding_h = padding[0];
    int64_t padding_w = padding[1];
    int64_t dilation_h = dilation[0];
    int64_t dilation_w = dilation[1];

    // Compute output dimensions
    int64_t H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int64_t W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    // Allocate output tensor
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Get pointers to input, weight, and bias (if available)
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    // Set up block and grid dimensions.
    // Each block handles one (n, c_out) pair and computes a TILE_H x TILE_W patch of the output spatial domain.
    dim3 block(TILE_W, TILE_H, 1);
    int tile_count_x = (W_out + TILE_W - 1) / TILE_W;
    int tile_count_y = (H_out + TILE_H - 1) / TILE_H;
    // Grid dimensions: x for batch, y for output channel, z for tiling the spatial domain
    dim3 grid(N, C_out, tile_count_x * tile_count_y);

    // Shared memory size for one weight filter (for one output channel), size in bytes
    int shmem_size = (C_in / groups) * K_h * K_w * sizeof(float);

    conv2d_cuda_kernel_shared_weight<<<grid, block, shmem_size>>>(
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
        printf("Error in conv2d_cuda_kernel_shared_weight: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda, "Optimized 2D convolution with shared weight (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}
