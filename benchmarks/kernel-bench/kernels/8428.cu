#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    bool bias_present) {

    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_weight[TILE_SIZE][TILE_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    if (idx >= N * C_out * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    float sum = 0.0f;

    for (int c = 0; c < C_in; c++) {
        for (int tile_h = 0; tile_h < kernel_h; tile_h += TILE_SIZE) {
            for (int tile_w = 0; tile_w < kernel_w; tile_w += TILE_SIZE) {
                int max_h = min(TILE_SIZE, kernel_h - tile_h);
                int max_w = min(TILE_SIZE, kernel_w - tile_w);

                // Load weight tile into shared memory
                if (tid < max_h * max_w) {
                    int h = tid / max_w;
                    int w = tid % max_w;
                    if (h < max_h && w < max_w) {
                        shared_weight[h][w] = weight[
                            c * (C_out * kernel_h * kernel_w) +
                            c_out * (kernel_h * kernel_w) +
                            (tile_h + h) * kernel_w +
                            (tile_w + w)];
                    }
                }
                __syncthreads();

                // Process the tile
                for (int i = 0; i < max_h; i++) {
                    for (int j = 0; j < max_w; j++) {
                        int h_diff = h_out + pad_h - (tile_h + i) * dilation_h;
                        int w_diff = w_out + pad_w - (tile_w + j) * dilation_w;

                        if (h_diff % stride_h == 0 && w_diff % stride_w == 0) {
                            int h_in = h_diff / stride_h;
                            int w_in = w_diff / stride_w;

                            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                float in_val = input[n * (C_in * H_in * W_in) +
                                                   c * (H_in * W_in) +
                                                   h_in * W_in + w_in];
                                sum += in_val * shared_weight[i][j];
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    if (bias_present) {
        sum += bias[c_out];
    }

    output[idx] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    auto N = x.size(0);
    auto C_in = x.size(1);
    auto H_in = x.size(2);
    auto W_in = x.size(3);
    auto C_out = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int H_out = (H_in - 1) * stride[0] - 2 * padding[0] + 
                dilation[0] * (kernel_h - 1) + output_padding[0] + 1;
    int W_out = (W_in - 1) * stride[1] - 2 * padding[1] + 
                dilation[1] * (kernel_w - 1) + output_padding[1] + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    int total_threads = N * C_out * H_out * W_out;
    int gridSize = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    bool bias_present = bias.has_value() && bias.value().defined();
    const float* bias_ptr = bias_present ? bias.value().data_ptr<float>() : nullptr;

    conv_transpose2d_kernel<<<gridSize, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        bias_present);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}