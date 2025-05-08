#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel using shared memory to cache weights and input tiles
__global__ void conv_transpose2d_kernel_shared(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int output_height,
    const int output_width,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w) {

    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + blockDim.x * blockDim.y;

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int num_threads = blockDim.x * gridDim.x;
    const int total_elements = batch_size * out_channels * output_height * output_width;

    for (int idx = tid; idx < total_elements; idx += num_threads) {
        const int w = idx % output_width;
        const int h = (idx / output_width) % output_height;
        const int c = (idx / (output_width * output_height)) % out_channels;
        const int b = idx / (output_width * output_height * out_channels);

        float sum = 0.0f;

        const int batch_offset = b * in_channels * input_height * input_width;
        const int out_ch_offset = c * kernel_height * kernel_width;

        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            const int in_ch_offset = in_ch * input_height * input_width;
            const int weight_in_offset = in_ch * out_channels * kernel_height * kernel_width;

            for (int kh = 0; kh < kernel_height; kh++) {
                const int in_y = h + pad_h - kh;
                if (in_y % stride_h != 0) continue;
                const int input_h = in_y / stride_h;
                if (input_h < 0 || input_h >= input_height) continue;

                for (int kw = 0; kw < kernel_width; kw++) {
                    const int in_x = w + pad_w - kw;
                    if (in_x % stride_w != 0) continue;
                    const int input_w = in_x / stride_w;
                    if (input_w < 0 || input_w >= input_width) continue;

                    // Load input and weight into shared memory
                    if (threadIdx.x < kernel_height * kernel_width) {
                        shared_weight[threadIdx.x] = weight[weight_in_offset + out_ch_offset + threadIdx.x];
                    }
                    if (threadIdx.x < blockDim.x * blockDim.y) {
                        shared_input[threadIdx.x] = input[batch_offset + in_ch_offset + input_h * input_width + input_w];
                    }
                    __syncthreads();

                    float input_val = shared_input[threadIdx.x];
                    float weight_val = shared_weight[threadIdx.x];
                    sum += input_val * weight_val;
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[c];
        }

        output[idx] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + 
                             kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + 
                            kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width},
                             x.options());

    const int block_size = 256;
    const int num_elements = batch_size * out_channels * output_height * output_width;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    const int grid_size = min(num_blocks, 65535);

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    size_t shared_mem_size = (blockDim.x * blockDim.y + kernel_height * kernel_width) * sizeof(float);

    conv_transpose2d_kernel_shared<<<grid_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "Shared memory ConvTranspose2D forward (CUDA)");
}
