#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function to calculate output position
__device__ __forceinline__ void calculate_output_position(
    const int idx, const int W_out, const int H_out, const int C_out,
    int& n, int& c_out, int& h_out, int& w_out
) {
    w_out = idx % W_out;
    int temp = idx / W_out;
    h_out = temp % H_out;
    temp /= H_out;
    c_out = temp % C_out;
    n = temp / C_out;
}

// Device function to calculate group information
__device__ __forceinline__ void calculate_group_info(
    const int c_out, const int C_out, const int C_in, const int groups,
    int& c_in_start, int& c_in_end
) {
    const int group = c_out / (C_out / groups);
    c_in_start = group * (C_in / groups);
    c_in_end = c_in_start + (C_in / groups);
}

// Device function to load input tile into shared memory
__device__ __forceinline__ void load_input_tile(
    const float* input,
    float* shared_input,
    const int n, const int c_in,
    const int h_start, const int w_start,
    const int H_in, const int W_in,
    const int tile_size
) {
    const int tid = threadIdx.x;
    const int input_offset = ((n * C_in + c_in) * H_in + h_start) * W_in + w_start;
    
    #pragma unroll
    for (int i = tid; i < tile_size * tile_size; i += blockDim.x) {
        const int tile_h = i / tile_size;
        const int tile_w = i % tile_size;
        const int h = h_start + tile_h;
        const int w = w_start + tile_w;
        
        if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
            shared_input[tile_h * tile_size + tile_w] = input[input_offset + tile_h * W_in + tile_w];
        } else {
            shared_input[tile_h * tile_size + tile_w] = 0.0f;
        }
    }
}

// Device function to load weight tile into shared memory
__device__ __forceinline__ void load_weight_tile(
    const float* weight,
    float* shared_weight,
    const int c_out, const int c_in_offset,
    const int K_h, const int K_w
) {
    const int tid = threadIdx.x;
    const int weight_offset = ((c_out * c_in_offset) * K_h) * K_w;
    
    #pragma unroll
    for (int i = tid; i < K_h * K_w; i += blockDim.x) {
        const int k_h = i / K_w;
        const int k_w = i % K_w;
        shared_weight[k_h * K_w + k_w] = weight[weight_offset + k_h * K_w + k_w];
    }
}

// Main convolution kernel
__global__ void conv2d_cuda_kernel_tiled(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int K_h, const int K_w,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int dilation_h, const int dilation_w,
    const int groups
) {
    extern __shared__ float shared_memory[];
    const int TILE_SIZE = 16;
    
    float* shared_input = shared_memory;
    float* shared_weight = shared_memory + TILE_SIZE * TILE_SIZE;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * H_out * W_out) return;

    int n, c_out, h_out, w_out;
    calculate_output_position(idx, W_out, H_out, C_out, n, c_out, h_out, w_out);

    int c_in_start, c_in_end;
    calculate_group_info(c_out, C_out, C_in, groups, c_in_start, c_in_end);
    
    const int C_in_per_group = C_in / groups;
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    const int h_in_start = h_out * stride_h - padding_h;
    const int w_in_start = w_out * stride_w - padding_w;

    for (int c_in = c_in_start; c_in < c_in_end; c_in += TILE_SIZE) {
        // Load input tile
        load_input_tile(input, shared_input, n, c_in,
                       h_in_start, w_in_start,
                       H_in, W_in, TILE_SIZE);
        
        // Load weight tile
        load_weight_tile(weight, shared_weight,
                        c_out, C_in_per_group,
                        K_h, K_w);
        
        __syncthreads();

        // Compute convolution for this tile
        #pragma unroll
        for (int k_h = 0; k_h < K_h; ++k_h) {
            #pragma unroll
            for (int k_w = 0; k_w < K_w; ++k_w) {
                const int h_in = h_in_start + k_h * dilation_h;
                const int w_in = w_in_start + k_w * dilation_w;
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    const int input_idx = (h_in - h_in_start) * TILE_SIZE + (w_in - w_in_start);
                    const int weight_idx = k_h * K_w + k_w;
                    value += shared_input[input_idx] * shared_weight[weight_idx];
                }
            }
        }
        
        __syncthreads();
    }

    output[idx] = value;
}

torch::Tensor conv2d_cuda_tiled(
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

    const auto stride_h = stride[0];
    const auto stride_w = stride[1];
    const auto padding_h = padding[0];
    const auto padding_w = padding[1];
    const auto dilation_h = dilation[0];
    const auto dilation_w = dilation[1];

    const auto H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    const auto W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    const int threads = 256;
    const int total_elements = N * C_out * H_out * W_out;
    const int blocks = (total_elements + threads - 1) / threads;
    const int shared_memory_size = (16 * 16 + K_h * K_w) * sizeof(float);

    conv2d_cuda_kernel_tiled<<<blocks, threads, shared_memory_size>>>(
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
    m.def("forward", &conv2d_cuda_tiled, "Tiled 2D convolution (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<int64_t>{1, 1},
        py::arg("padding") = std::vector<int64_t>{0, 0},
        py::arg("dilation") = std::vector<int64_t>{1, 1},
        py::arg("groups") = 1
    );
}