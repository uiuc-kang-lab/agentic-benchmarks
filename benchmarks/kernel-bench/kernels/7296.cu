#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Shared memory tile sizes
#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 11

__global__ void conv2d_cuda_kernel_hybrid(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int chunk_N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + (TILE_SIZE + K_h - 1) * (TILE_SIZE + K_w - 1);

    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.y % TILE_SIZE;
    int c_out_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    
    if (batch_idx >= chunk_N) return;

    // Calculate output coordinates
    int h_out_base = (blockIdx.z / ((W_out + TILE_SIZE - 1) / TILE_SIZE)) * TILE_SIZE;
    int w_out_base = (blockIdx.z % ((W_out + TILE_SIZE - 1) / TILE_SIZE)) * TILE_SIZE;
    int h_out = h_out_base + ty;
    int w_out = w_out_base + tx;

    float sum = (bias != nullptr) ? bias[c_out_idx] : 0.0f;

    int group = c_out_idx / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);

    // Load kernel weights into shared memory
    if (ty < K_h && tx < K_w) {
        for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
            int weight_idx = (((c_out_idx * (C_in / groups) + (c_in - c_in_start)) * K_h + ty) * K_w) + tx;
            shared_weight[ty * K_w + tx] = weight[weight_idx];
        }
    }
    __syncthreads();

    if (h_out < H_out && w_out < W_out) {
        for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
            // Load input tile into shared memory
            int h_in_base = h_out * stride_h - padding_h;
            int w_in_base = w_out * stride_w - padding_w;
            
            for (int i = ty; i < TILE_SIZE + K_h - 1; i += TILE_SIZE) {
                for (int j = tx; j < TILE_SIZE + K_w - 1; j += TILE_SIZE) {
                    int h_in = h_in_base + i;
                    int w_in = w_in_base + j;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        shared_input[i * (TILE_SIZE + K_w - 1) + j] = 
                            input[((batch_idx * C_in + c_in) * H_in + h_in) * W_in + w_in];
                    } else {
                        shared_input[i * (TILE_SIZE + K_w - 1) + j] = 0.0f;
                    }
                }
            }
            __syncthreads();

            // Compute convolution using shared memory
            for (int k_h = 0; k_h < K_h; ++k_h) {
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    int h_offset = ty * stride_h + k_h * dilation_h;
                    int w_offset = tx * stride_w + k_w * dilation_w;
                    sum += shared_input[h_offset * (TILE_SIZE + K_w - 1) + w_offset] *
                           shared_weight[k_h * K_w + k_w];
                }
            }
            __syncthreads();
        }

        if (h_out < H_out && w_out < W_out) {
            int output_idx = ((batch_idx * C_out + c_out_idx) * H_out + h_out) * W_out + w_out;
            output[output_idx] = sum;
        }
    }
}

torch::Tensor conv2d_cuda_hybrid(
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

    const auto N = input.size(0);
    const auto C_in = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);
    const auto C_out = weight.size(0);
    const auto K_h = weight.size(2);
    const auto K_w = weight.size(3);

    const auto stride_h = stride[0];
    const auto stride_w = stride[1];
    const auto padding_h = padding[0];
    const auto padding_w = padding[1];
    const auto dilation_h = dilation[0];
    const auto dilation_w = dilation[1];

    const auto H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    const auto W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const int num_streams = std::min(4, static_cast<int>(N));
    const int chunk_size = (N + num_streams - 1) / num_streams;

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 block(TILE_SIZE, TILE_SIZE);
    size_t shared_mem_size = ((TILE_SIZE + K_h - 1) * (TILE_SIZE + K_w - 1) + K_h * K_w) * sizeof(float);

    for (int i = 0; i < num_streams; ++i) {
        int batch_start = i * chunk_size;
        if (batch_start >= N) break;
        
        int chunk_N = std::min(chunk_size, static_cast<int>(N - batch_start));
        
        dim3 grid(C_out, 
                 chunk_N,
                 ((H_out + TILE_SIZE - 1) / TILE_SIZE) * ((W_out + TILE_SIZE - 1) / TILE_SIZE));

        const float* input_ptr = input.data_ptr<float>() + batch_start * C_in * H_in * W_in;
        float* output_ptr = output.data_ptr<float>() + batch_start * C_out * H_out * W_out;
        const float* weight_ptr = weight.data_ptr<float>();
        const float* bias_ptr = bias_opt.has_value() ? bias_opt.value().data_ptr<float>() : nullptr;

        conv2d_cuda_kernel_hybrid<<<grid, block, shared_mem_size, streams[i]>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            chunk_N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            K_h, K_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups
        );
    }

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}