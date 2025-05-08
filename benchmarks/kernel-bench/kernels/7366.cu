#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_SHARED_ELEMENTS 2048

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
    extern __shared__ float shared_mem[];
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C_out * H_out * W_out;
    
    if (index >= total_threads) return;
    
    int w_out = index % W_out;
    int tmp = index / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;
    
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);
    
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    float partial_sum = 0.0f;
    
    // Compute convolution
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int k_h = 0; k_h < K_h; ++k_h) {
            for (int k_w = 0; k_w < K_w; ++k_w) {
                int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
                int w_in = w_out * stride_w - padding_w + k_w * dilation_w;
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int weight_idx = ((c_out * (C_in / groups) + (c_in - c_in_start)) * K_h + k_h) * K_w + k_w;
                    
                    partial_sum += input[input_idx] * weight[weight_idx];
                }
            }
            }
        }
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }
    
    // First thread in each warp writes result
    if (lane_id == 0) {
        shared_mem[warp_id] = partial_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (threadIdx.x < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        partial_sum = shared_mem[threadIdx.x];
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        value += partial_sum;
        int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
        output[output_idx] = value;
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
    
    auto dims = input.sizes();
    int64_t N = dims[0];
    int64_t C_in = dims[1];
    int64_t H_in = dims[2];
    int64_t W_in = dims[3];
    
    auto w_dims = weight.sizes();
    int64_t C_out = w_dims[0];
    int64_t K_h = w_dims[2];
    int64_t K_w = w_dims[3];
    
    int64_t H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) / stride[0] + 1;
    int64_t W_out = (W_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) / stride[1] + 1;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        bias_ptr = bias_opt.value().contiguous().data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();
    
    int total_threads = N * C_out * H_out * W_out;
    int threads_per_block = BLOCK_SIZE;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    int shared_mem_size = MAX_SHARED_ELEMENTS * sizeof(float);
    
    conv2d_cuda_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
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