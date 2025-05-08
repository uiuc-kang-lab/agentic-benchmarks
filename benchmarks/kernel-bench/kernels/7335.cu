#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define tile sizes for better cache utilization
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define BLOCK_SIZE 256

__global__ void conv2d_cuda_kernel(
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
    __shared__ float input_tile[TILE_HEIGHT][TILE_WIDTH];
    __shared__ float weight_tile[TILE_HEIGHT][TILE_WIDTH];

    // Calculate output position
    int batch_idx = blockIdx.z / ((C_out + TILE_HEIGHT - 1) / TILE_HEIGHT);
    int out_channel_block = blockIdx.z % ((C_out + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    int h_out_start = blockIdx.y * TILE_HEIGHT;
    int w_out_start = blockIdx.x * TILE_WIDTH;
    
    int thread_id = threadIdx.x;
    int thread_row = thread_id / TILE_WIDTH;
    int thread_col = thread_id % TILE_WIDTH;

    // Pre-calculate channel ranges for groups
    int channels_per_group = C_in / groups;
    int out_channels_per_group = C_out / groups;
    
    float local_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Each thread handles multiple output elements
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        int h_out = h_out_start + thread_row + i * (TILE_HEIGHT/4);
        int w_out = w_out_start + thread_col;
        
        if (h_out >= H_out || w_out >= W_out) continue;
        
        // Calculate group and channel ranges
        int out_channel_start = out_channel_block * TILE_HEIGHT;
        int group = out_channel_start / out_channels_per_group;
        int c_in_start = group * channels_per_group;
        int c_in_end = c_in_start + channels_per_group;

        // Process input channels in tiles
        for (int c_in = c_in_start; c_in < c_in_end; c_in += TILE_WIDTH) {
            // Compute convolution for each kernel position
            for (int k_h = 0; k_h < K_h; k_h++) {
                for (int k_w = 0; k_w < K_w; k_w++) {
                    int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
                    int w_in = w_out * stride_w - padding_w + k_w * dilation_w;
                    
                    // Load input tile cooperatively
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        for (int c = 0; c < min(TILE_WIDTH, c_in_end - c_in); c++) {
                            if (thread_id < TILE_WIDTH) {
                                int input_idx = ((batch_idx * C_in + (c_in + c)) * H_in + h_in) * W_in + w_in;
                                input_tile[thread_id][c] = input[input_idx];
                            }
                        }
                    } else {
                        for (int c = 0; c < min(TILE_WIDTH, c_in_end - c_in); c++) {
                            if (thread_id < TILE_WIDTH) {
                                input_tile[thread_id][c] = 0.0f;
                            }
                        }
                    }

                    // Load weight tile cooperatively
                    if (thread_id < TILE_WIDTH) {
                        int out_channel = out_channel_start + thread_row;
                        if (out_channel < C_out) {
                            for (int c = 0; c < min(TILE_WIDTH, c_in_end - c_in); c++) {
                                int weight_idx = ((out_channel * channels_per_group + (c_in + c - c_in_start)) * K_h + k_h) * K_w + k_w;
                                weight_tile[thread_id][c] = weight[weight_idx];
                            }
                        }
                    }
                    
                    __syncthreads();
                    
                    // Process channels
                    if (h_out < H_out && w_out < W_out) {
                        int out_channel = out_channel_start + thread_row;
                        if (out_channel < C_out) {
                            for (int c = 0; c < min(TILE_WIDTH, c_in_end - c_in); c++) {
                                local_sum[i] += input_tile[thread_row][c] * weight_tile[thread_row][c];
                            }
                        }
                    }
                    
                    __syncthreads();
                }
            }
        }
        
        // Write output
        if (h_out < H_out && w_out < W_out) {
            int out_channel = out_channel_start + thread_row;
            if (out_channel < C_out) {
                int output_idx = ((batch_idx * C_out + out_channel) * H_out + h_out) * W_out + w_out;
                float bias_val = (bias != nullptr) ? bias[out_channel] : 0.0f;
                output[output_idx] = local_sum[i] + bias_val;
            }
        }
    }
}

torch::Tensor conv2d_cuda(
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

    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    auto C_out = weight.size(0);
    auto K_h = weight.size(2);
    auto K_w = weight.size(3);

    auto stride_h = stride[0];
    auto stride_w = stride[1];
    auto padding_h = padding[0];
    auto padding_w = padding[1];
    auto dilation_h = dilation[0];
    auto dilation_w = dilation[1];

    auto H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    auto W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        bias_ptr = bias_opt.value().contiguous().data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    // Calculate grid dimensions for tiled approach
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(
        (W_out + TILE_WIDTH - 1) / TILE_WIDTH,
        (H_out + TILE_HEIGHT - 1) / TILE_HEIGHT,
        N * ((C_out + TILE_HEIGHT - 1) / TILE_HEIGHT)
    );

    conv2d_cuda_kernel<<<blocks, threads>>>(
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

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda, "Custom 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}