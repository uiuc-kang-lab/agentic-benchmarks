#include <torch/extension.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding) {

    extern __shared__ float shared_mem[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Load input and weight data into shared memory with aligned access
    if (tx < kernel_size && ty < kernel_size) {
        for (int c = 0; c < in_channels; c++) {
            shared_mem[c * kernel_size * kernel_size + ty * kernel_size + tx] =
                __ldg(&weight[c * kernel_size * kernel_size + ty * kernel_size + tx]);
        }
    }
    __syncthreads();

    const int out_y = by * blockDim.y + ty;
    const int out_x = bx * blockDim.x + tx;

    if (out_y < out_height && out_x < out_width) {
        for (int oc = 0; oc < out_channels; oc++) {
            float sum = bias ? __ldg(&bias[oc]) : 0.0f;

            for (int ic = 0; ic < in_channels; ic++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        const int in_y = (out_y + padding - ky) / stride;
                        const int in_x = (out_x + padding - kx) / stride;

                        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                            const float in_val = __ldg(&input[((bz * in_channels + ic) *
                                in_height + in_y) * in_width + in_x]);
                            const float weight_val = shared_mem[ic * kernel_size * kernel_size +
                                ky * kernel_size + kx];
                            sum += in_val * weight_val;
                        }
                    }
                }
            }

            output[((bz * out_channels + oc) * out_height + out_y) * out_width + out_x] = sum;
        }
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    auto input_sizes = x.sizes();
    auto weight_sizes = weight.sizes();
    
    const int batch_size = input_sizes[0];
    const int in_channels = input_sizes[1];
    const int in_height = input_sizes[2];
    const int in_width = input_sizes[3];
    const int kernel_size = weight_sizes[2];
    const int out_channels = weight_sizes[1];

    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
        x.options());

    const dim3 threads(16, 16);
    const dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size
    );

    const int shared_mem_size = in_channels * kernel_size * kernel_size * sizeof(float);

    conv_transpose2d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        output_padding
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}