#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile size for shared memory optimization
#define TILE_SIZE 16
#define BLOCK_SIZE 16

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int N, const int C_in, const int H, const int W,
    const int C_out, const int K,
    const int stride, const int padding,
    const int H_out, const int W_out) {
    
    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_weight[TILE_SIZE][TILE_SIZE];
    
    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Calculate output position
    const int out_x = bx * BLOCK_SIZE + tx;
    const int out_y = by * BLOCK_SIZE + ty;
    
    // Batch and channel indices
    const int n = bz / C_out;
    const int oc = bz % C_out;
    
    if (n >= N || out_x >= W_out || out_y >= H_out) return;
    
    float acc = 0.0f;
    
    // Calculate input boundaries
    const int in_x_start = (out_x + padding) / stride;
    const int in_y_start = (out_y + padding) / stride;
    const int in_x_end = min((out_x + padding + K - 1) / stride + 1, W);
    const int in_y_end = min((out_y + padding + K - 1) / stride + 1, H);
    
    // Process input channels
    for (int ic = 0; ic < C_in; ic++) {
        for (int in_y = in_y_start; in_y < in_y_end; in_y++) {
            for (int in_x = in_x_start; in_x < in_x_end; in_x++) {
                // Load input tile to shared memory
                if (tx < (in_x_end - in_x_start) && ty < (in_y_end - in_y_start)) {
                    shared_input[ty][tx] = input[
                        ((n * C_in + ic) * H + in_y) * W + in_x
                    ];
                }
                
                // Load weight tile to shared memory
                const int k_y = out_y + padding - in_y * stride;
                const int k_x = out_x + padding - in_x * stride;
                if (k_y >= 0 && k_y < K && k_x >= 0 && k_x < K) {
                    shared_weight[ty][tx] = weight[
                        ((ic * C_out + oc) * K + k_y) * K + k_x
                    ];
                }
                
                __syncthreads();
                
                // Compute partial sum
                if (tx < (in_x_end - in_x_start) && ty < (in_y_end - in_y_start)) {
                    acc += shared_input[ty][tx] * shared_weight[ty][tx];
                }
                
                __syncthreads();
            }
        }
    }
    
    // Write result to global memory
    if (out_x < W_out && out_y < H_out) {
        const int out_idx = ((n * C_out + oc) * H_out + out_y) * W_out + out_x;
        output[out_idx] = acc;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    
    const auto N = x.size(0);
    const auto C_in = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    const auto C_out = weight.size(1);
    const auto K = weight.size(2);
    
    const auto H_out = (H - 1) * stride - 2 * padding + K + output_padding;
    const auto W_out = (W - 1) * stride - 2 * padding + K + output_padding;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (H_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
        N * C_out
    );
    
    conv_transpose2d_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out, K,
        stride, padding,
        H_out, W_out
    );
    
    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}