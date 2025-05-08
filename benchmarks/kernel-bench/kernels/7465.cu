#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 11

__global__ void conv_transpose2d_kernel_shared(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding) {

    extern __shared__ float shared_weight[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int oc = blockIdx.z % C_out;
    const int b = blockIdx.z / C_out;

    // Load weights into shared memory
    if (tx < K && ty < K) {
        for (int ic = 0; ic < C_in; ++ic) {
            shared_weight[ic * K * K + ty * K + tx] = 
                weight[ic * (C_out * K * K) + oc * (K * K) + ty * K + tx];
        }
    }
    __syncthreads();

    const int h = by + ty;
    const int w = bx + tx;

    if (h < H_out && w < W_out) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < C_in; ++ic) {
            for (int kh = 0; kh < K; ++kh) {
                int h_in_candidate = h + padding - kh;
                if (h_in_candidate % stride == 0) {
                    int h_in = h_in_candidate / stride;
                    if (h_in >= 0 && h_in < H_in) {
                        for (int kw = 0; kw < K; ++kw) {
                            int w_in_candidate = w + padding - kw;
                            
                            if (w_in_candidate % stride == 0) {
                                int w_in = w_in_candidate / stride;
                                
                                if (w_in >= 0 && w_in < W_in) {
                                    int input_idx = b * (C_in * H_in * W_in) + ic * (H_in * W_in) + h_in * W_in + w_in;
                                    float input_val = input[input_idx];
                                    float weight_val = shared_weight[ic * K * K + kh * K + kw];
                                    sum += input_val * weight_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[oc];
        }

        int out_idx = b * (C_out * H_out * W_out) + oc * (H_out * W_out) + h * W_out + w;
        output[out_idx] = sum;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    TORCH_CHECK(groups == 1, "Only groups==1 is supported");
    TORCH_CHECK(output_padding == 0, "Only output_padding==0 is supported");

    int B = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int K = weight.size(2);
    int C_out = weight.size(1);
    int H_out = (H_in - 1) * stride - 2 * padding + K;
    int W_out = (W_in - 1) * stride - 2 * padding + K;

    auto output_tensor = torch::zeros({B, C_out, H_out, W_out}, input.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (W_out + TILE_SIZE - 1) / TILE_SIZE,
        (H_out + TILE_SIZE - 1) / TILE_SIZE,
        B * C_out
    );

    int shared_mem_size = C_in * K * K * sizeof(float);

    conv_transpose2d_kernel_shared<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output_tensor.data_ptr<float>(),
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding);

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}