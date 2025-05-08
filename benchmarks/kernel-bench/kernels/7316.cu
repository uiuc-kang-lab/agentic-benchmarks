#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <int BLOCK_SIZE = 256>
__global__ void conv2d_cuda_kernel_shared_memory(
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
    float* shared_weight = &shared_mem[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int index = blockIdx.x * blockDim.x + tid;
    const int total_elements = N * C_out * H_out * W_out;
    
    if (index >= total_elements) return;

    // Calculate output position
    const int w_out = index % W_out;
    int temp = index / W_out;
    const int h_out = temp % H_out;
    temp /= H_out;
    const int c_out = temp % C_out;
    const int n = temp / C_out;

    // Calculate group information
    const int group = c_out / (C_out / groups);
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);
    const int C_in_per_group = C_in / groups;

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Pre-calculate input base positions
    const int h_in_base = h_out * stride_h - padding_h;
    const int w_in_base = w_out * stride_w - padding_w;

    // Process input channels in chunks to utilize shared memory
    const int chunk_size = BLOCK_SIZE / (K_h * K_w);
    
    for (int c_in = c_in_start; c_in < c_in_end; c_in += chunk_size) {
        const int current_chunk_size = min(chunk_size, c_in_end - c_in);
        
        // Load weights into shared memory
        if (tid < current_chunk_size * K_h * K_w) {
            const int weight_idx = ((c_out * C_in_per_group + (c_in - c_in_start)) * K_h * K_w) + tid;
            shared_weight[tid] = weight[weight_idx];
        }
        
        // Only synchronize after weight loading if we're using the shared memory
        if (current_chunk_size > 0) {
            __syncthreads();
        }

        // Process the chunk
        for (int ci = 0; ci < current_chunk_size; ++ci) {
            const int current_c_in = c_in + ci;
            const int input_n_offset = n * C_in * H_in * W_in;
            const int input_c_offset = current_c_in * H_in * W_in;

            for (int kh = 0; kh < K_h; ++kh) {
                const int h_in = h_in_base + kh * dilation_h;
                
                if (h_in >= 0 && h_in < H_in) {
                    for (int kw = 0; kw < K_w; ++kw) {
                        const int w_in = w_in_base + kw * dilation_w;
                        
                        if (w_in >= 0 && w_in < W_in) {
                            const int input_idx = input_n_offset + input_c_offset + h_in * W_in + w_in;
                            const int weight_idx = ci * K_h * K_w + kh * K_w + kw;
                            value += input[input_idx] * shared_weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Only synchronize if there's more data to process
        if (c_in + chunk_size < c_in_end) {
            __syncthreads();
        }
    }

    output[index] = value;
}

torch::Tensor conv2d_cuda_shared_memory(
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

    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int C_out = weight.size(0);
    const int K_h = weight.size(2);
    const int K_w = weight.size(3);

    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int padding_h = padding[0];
    const int padding_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];

    const int H_out = (H_in + 2 * padding_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    const int W_out = (W_in + 2 * padding_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value().contiguous();
        bias_ptr = bias.data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    constexpr int BLOCK_SIZE = 256;
    const int total_elements = N * C_out * H_out * W_out;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Calculate shared memory size
    const int shared_mem_size = BLOCK_SIZE * sizeof(float) * 2; // For input and weight data

    conv2d_cuda_kernel_shared_memory<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
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
    m.def("forward", &conv2d_cuda_shared_memory, "Shared memory 2D convolution with minimal sync (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0},
          py::arg("dilation") = std::vector<int64_t>{1, 1},
          py::arg("groups") = 1);
}