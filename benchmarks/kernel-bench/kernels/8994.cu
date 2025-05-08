#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

__global__ void conv1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation
) {
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_idx = blockIdx.x * warps_per_block + warp_idx;
    
    const int total_elements = B * out_channels * out_size;
    const int total_warps = (total_elements + WARP_SIZE - 1) / WARP_SIZE;
    
    if (global_warp_idx >= total_warps) return;

    const int base_idx = global_warp_idx * WARP_SIZE + lane_idx;
    
    if ((base_idx - lane_idx + WARP_SIZE) <= total_elements) {
        const int o = base_idx % out_size;
        const int tmp = base_idx / out_size;
        const int oc = tmp % out_channels;
        const int b = tmp / out_channels;

        const int start_pos = o * stride;
        const int valid = (base_idx < total_elements);
        
        float sum = 0.0f;
        
        if (valid) {
            for (int ic = 0; ic < in_channels; ++ic) {
                const float* x_base = x + b * (in_channels * in_size) + ic * in_size;
                const float* w_base = weight + oc * (in_channels * kernel_size) + ic * kernel_size;

                #pragma unroll 4
                for (int k = 0; k < kernel_size; ++k) {
                    const int input_pos = start_pos + k * dilation;
                    const float x_val = (input_pos < in_size) ? x_base[input_pos] : 0.0f;
                    sum += x_val * w_base[k];
                }
            }

            if (bias != nullptr) {
                sum += bias[oc];
            }
            
            output[b * (out_channels * out_size) + oc * out_size + o] = sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels mismatch");

    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias->size(0) == weight.size(0), "Bias size mismatch");
    }

    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias ? bias->data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    const int threads_per_block = 256;
    const int total_elements = B * out_channels * out_size;
    const int warps_needed = (total_elements + WARP_SIZE - 1) / WARP_SIZE;
    const int blocks = (warps_needed + (threads_per_block / WARP_SIZE) - 1) / (threads_per_block / WARP_SIZE);

    conv1d_kernel<<<blocks, threads_per_block>>>(
        x_data,
        weight_data,
        bias_data,
        output_data,
        B,
        in_channels,
        in_size,
        out_channels,
        kernel_size,
        out_size,
        stride,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward (CUDA) with warp-aligned processing");
}