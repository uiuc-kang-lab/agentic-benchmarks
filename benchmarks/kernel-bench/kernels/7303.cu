#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Shared memory tile dimensions
#define TILE_SIZE 16
#define BLOCK_SIZE 256

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
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + TILE_SIZE * TILE_SIZE;

    const int tid = threadIdx.x;
    const int output_idx = blockIdx.x * blockDim.x + tid;
    
    if (output_idx >= N * C_out * H_out * W_out) return;

    // Calculate output position
    const int w_out = output_idx % W_out;
    int tmp = output_idx / W_out;
    const int h_out = tmp % H_out;
    tmp = tmp / H_out;
    const int c_out = tmp % C_out;
    const int n = tmp / C_out;

    // Calculate group information
    const int group = c_out / (C_out / groups);
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);
    const int C_in_per_group = C_in / groups;

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Input window boundaries
    const int h_in_start = h_out * stride_h - padding_h;
    const int w_in_start = w_out * stride_w - padding_w;

    // Process input channels in tiles
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        // Load kernel weights into shared memory
        const int weight_tile_size = K_h * K_w;
        for (int k = tid; k < weight_tile_size; k += blockDim.x) {
            const int k_h = k / K_w;
            const int k_w = k % K_w;
            const int weight_idx = (((c_out * C_in_per_group + (c_in - c_in_start)) * K_h + k_h) * K_w) + k_w;
            shared_weight[k] = weight[weight_idx];
        }

        // Load input tile into shared memory
        const int input_tile_size = (K_h + TILE_SIZE - 1) * (K_w + TILE_SIZE - 1);
        for (int i = tid; i < input_tile_size; i += blockDim.x) {
            const int tile_h = i / (K_w + TILE_SIZE - 1);
            const int tile_w = i % (K_w + TILE_SIZE - 1);
            const int h_in = h_in_start + tile_h;
            const int w_in = w_in_start + tile_w;
            
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                const int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                shared_input[tile_h * TILE_SIZE + tile_w] = input[input_idx];
            } else {
                shared_input[tile_h * TILE_SIZE + tile_w] = 0.0f;
            }
        }
        
        // Ensure shared memory is loaded
        __syncthreads();

        // Compute convolution using shared memory
        #pragma unroll
        for (int k_h = 0; k_h < K_h; ++k_h) {
            #pragma unroll
            for (int k_w = 0; k_w < K_w; ++k_w) {
                const int h_in = h_in_start + k_h * dilation_h;
                const int w_in = w_in_start + k_w * dilation_w;
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    const int shared_h = k_h;
                    const int shared_w = k_w;
                    value += shared_input[shared_h * TILE_SIZE + shared_w] * 
                            shared_weight[k_h * K_w + k_w];
                }
            }
        }

        // Synchronize before next iteration
        __syncthreads();
    }

    // Write output
    if (output_idx < N * C_out * H_out * W_out) {
        output[output_idx] = value;
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

    const auto N = input.size(0);
    const auto C_in = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);
    const auto C_out = weight.size(0);
    const auto K_h = weight.size(2);
    const auto K_w = weight.size(3);

    const auto H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) / stride[0] + 1;
    const auto W_out = (W_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) / stride[1] + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    const int total_threads = N * C_out * H_out * W_out;
    const int threads_per_block = BLOCK_SIZE;
    const int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    // Calculate shared memory size
    const int shared_mem_size = (TILE_SIZE * TILE_SIZE + K_h * K_w) * sizeof(float);

    conv2d_cuda_kernel_shared<<<num_blocks, threads_per_block, shared_mem_size>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cuda_shared, "Shared memory 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}