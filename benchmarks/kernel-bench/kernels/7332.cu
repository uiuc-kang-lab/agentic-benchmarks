#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <int TILE_SIZE = 16, int WARP_SIZE = 32>
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
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + TILE_SIZE * TILE_SIZE;

    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    const int h_out = blockIdx.y;
    const int w_out_base = blockIdx.x * TILE_SIZE;
    const int c_out = blockIdx.z % C_out;
    const int n = blockIdx.z / C_out;

    const int group = c_out / (C_out / groups);
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);

    float partial_sum = 0.0f;
    
    for (int c_in = c_in_start; c_in < c_in_end; c_in += TILE_SIZE) {
        #pragma unroll
        for (int k_h = 0; k_h < K_h; ++k_h) {
            const int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
            
            if (h_in >= 0 && h_in < H_in) {
                for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
                    const int tile_w = i % TILE_SIZE;
                    const int c_in_offset = i / TILE_SIZE;
                    const int w_in = (w_out_base + tile_w) * stride_w - padding_w;
                    
                    if (c_in + c_in_offset < c_in_end && w_in >= 0 && w_in < W_in) {
                        shared_input[i] = input[((n * C_in + (c_in + c_in_offset)) * H_in + h_in) * W_in + w_in];
                    } else {
                        shared_input[i] = 0.0f;
                    }
                }

                for (int i = tid; i < TILE_SIZE * K_w; i += blockDim.x) {
                    const int k_w = i % K_w;
                    const int c_in_offset = i / K_w;
                    if (c_in + c_in_offset < c_in_end) {
                        shared_weight[i] = weight[(((c_out * (C_in / groups) + (c_in + c_in_offset - c_in_start)) * K_h + k_h) * K_w) + k_w];
                    } else {
                        shared_weight[i] = 0.0f;
                    }
                }

                __syncthreads();

                if (w_out_base + tid < W_out) {
                    #pragma unroll
                    for (int tile_idx = 0; tile_idx < TILE_SIZE && (c_in + tile_idx) < c_in_end; ++tile_idx) {
                        #pragma unroll
                        for (int k_w = 0; k_w < K_w; ++k_w) {
                            const int w_in_idx = tid * stride_w + k_w * dilation_w;
                            if (w_in_idx >= 0 && w_in_idx < TILE_SIZE) {
                                partial_sum += shared_input[tile_idx * TILE_SIZE + w_in_idx] *
                                             shared_weight[tile_idx * K_w + k_w];
                            }
                        }
                    }
                }

                __syncthreads();
            }
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    if (lane == 0 && w_out_base + wid < W_out) {
        float final_value = partial_sum;
        if (bias != nullptr && tid == 0) {
            final_value += bias[c_out];
        }
        output[((n * C_out + c_out) * H_out + h_out) * W_out + (w_out_base + wid)] = final_value;
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

    const int TILE_SIZE = 16;
    const int BLOCK_SIZE = 128;

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

    dim3 threads(BLOCK_SIZE);
    dim3 blocks(
        (W_out + TILE_SIZE - 1) / TILE_SIZE,
        H_out,
        N * C_out
    );

    int shared_mem_size = (TILE_SIZE * TILE_SIZE + TILE_SIZE * K_w) * sizeof(float);

    conv2d_cuda_kernel<TILE_SIZE><<<blocks, threads, shared_mem_size>>>(
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