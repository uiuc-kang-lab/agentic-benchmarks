#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define shared memory tile dimensions
#define TILE_DIM 16
#define BLOCK_ROWS 8

__global__ void conv2d_cuda_kernel_shared(
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
    __shared__ float shared_input[TILE_DIM][TILE_DIM + 1];  // +1 for bank conflicts
    __shared__ float shared_weight[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int n = bz / C_out;
    const int c_out = bz % C_out;
    
    const int h_out_start = by * BLOCK_ROWS;
    const int w_out = bx * TILE_DIM + tx;

    const int group = c_out / (C_out / groups);
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Process input channels in tiles
    for (int c_in = c_in_start; c_in < c_in_end; c_in++) {
        for (int tile = 0; tile < (K_h * K_w + TILE_DIM - 1) / TILE_DIM; tile++) {
            const int kernel_idx_start = tile * TILE_DIM;
            
            // Load weight tile into shared memory
            if (kernel_idx_start + tx < K_h * K_w && ty < TILE_DIM) {
                const int weight_idx = ((c_out * (C_in / groups) + (c_in - c_in_start)) * K_h * K_w) + kernel_idx_start + tx;
                shared_weight[ty][tx] = (kernel_idx_start + tx < K_h * K_w) ? weight[weight_idx] : 0.0f;
            }

            // Only synchronize after shared memory writes
            __syncthreads();

            for (int h_offset = 0; h_offset < BLOCK_ROWS; h_offset++) {
                const int h_out = h_out_start + h_offset;
                if (h_out >= H_out || w_out >= W_out) continue;

                // Load input tile into shared memory
                const int h_in_base = h_out * stride_h - padding_h;
                const int w_in_base = w_out * stride_w - padding_w;

                for (int k = 0; k < TILE_DIM; k++) {
                    if (kernel_idx_start + k >= K_h * K_w) break;
                    
                    const int k_h = (kernel_idx_start + k) / K_w;
                    const int k_w = (kernel_idx_start + k) % K_w;
                    
                    const int h_in = h_in_base + k_h * dilation_h;
                    const int w_in = w_in_base + k_w * dilation_w;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        const int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                        value += input[input_idx] * shared_weight[ty][k];
                    }
                }
            }

            // Only synchronize before next iteration if there is one
            if (tile < (K_h * K_w + TILE_DIM - 1) / TILE_DIM - 1) {
                __syncthreads();
            }
        }
    }

    // Write output
    for (int h_offset = 0; h_offset < BLOCK_ROWS; h_offset++) {
        const int h_out = h_out_start + h_offset;
        if (h_out < H_out && w_out < W_out) {
            const int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
            output[output_idx] = value;
        }
    }
}

torch::Tensor conv2d_cuda_shared(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    input = input.contiguous();
    weight = weight.contiguous();

    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");

    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "Bias tensor must be on CUDA if provided");
    }

    const int64_t N = input.size(0);
    const int64_t C_in = input.size(1);
    const int64_t H_in = input.size(2);
    const int64_t W_in = input.size(3);
    const int64_t C_out = weight.size(0);
    const int64_t K_h = weight.size(2);
    const int64_t K_w = weight.size(3);

    const int64_t stride_h = stride[0];
    const int64_t stride_w = stride[1];
    const int64_t padding_h = padding[0];
    const int64_t padding_w = padding[1];
    const int64_t dilation_h = dilation[0];
    const int64_t dilation_w = dilation[1];

    const int64_t H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    const int64_t W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    // Configure kernel launch parameters
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks(
        (W_out + TILE_DIM - 1) / TILE_DIM,
        (H_out + BLOCK_ROWS - 1) / BLOCK_ROWS,
        N * C_out
    );

    conv2d_cuda_kernel_shared<<<blocks, threads>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_shared, "Shared memory optimized 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}