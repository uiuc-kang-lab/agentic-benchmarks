#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int B,
    const int in_channels,
    const int in_size,
    const int out_channels,
    const int kernel_size,
    const int out_size,
    const int stride,
    const int dilation
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = gridDim.x;
    
    const int total_elements = B * out_channels * out_size;
    const int elements_per_grid = (total_elements + grid_size - 1) / grid_size;
    const int start_idx = bid * elements_per_grid;
    const int end_idx = min(start_idx + elements_per_grid, total_elements);

    for (int base_idx = start_idx + tid; base_idx < end_idx; base_idx += block_size) {
        const int o = base_idx % out_size;
        const int temp = base_idx / out_size;
        const int oc = temp % out_channels;
        const int b = temp / out_channels;

        const int input_start = o * stride;
        const float* x_batch = x + b * (in_channels * in_size);
        const float* w_oc = weight + oc * (in_channels * kernel_size);
        
        float sum = 0.0f;
        
        #pragma unroll 2
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* x_ic = x_batch + ic * in_size + input_start;
            const float* w_ic = w_oc + ic * kernel_size;
            
            if (input_start + (kernel_size - 1) * dilation < in_size) {
                #pragma unroll 4
                for (int k = 0; k < kernel_size; ++k) {
                    sum += x_ic[k * dilation] * w_ic[k];
                }
            } else {
                #pragma unroll 4
                for (int k = 0; k < kernel_size; ++k) {
                    const int pos = k * dilation;
                    if (input_start + pos < in_size) {
                        sum += x_ic[pos] * w_ic[k];
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[oc];
        }

        output[base_idx] = sum;
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

    const int threads = 256;
    const int max_blocks = 256;
    const int total_elements = B * out_channels * out_size;
    const int blocks = min(max_blocks, (total_elements + threads - 1) / threads);

    conv1d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "1D convolution forward (CUDA) with stride loop optimization");
}