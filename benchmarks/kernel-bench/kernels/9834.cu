#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined kernel using warp-level reduction and shared memory optimization
__global__ void depthwise_conv2d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group,
    int total_output
) {
    extern __shared__ float shared_data[];
    const int warpSize = 32;
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = globalThreadId / warpSize;
    int lane = globalThreadId % warpSize;

    if (warpId >= total_output) return;

    int tmp = warpId;
    int out_w_idx = tmp % output_w;
    tmp /= output_w;
    int out_h_idx = tmp % output_h;
    tmp /= output_h;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    int in_ch = oc / channels_per_group;

    int in_y_origin = out_h_idx * stride - padding;
    int in_x_origin = out_w_idx * stride - padding;

    int kernel_area = kernel_size * kernel_size;
    float sum = 0.0f;
    
    for (int i = lane; i < kernel_area; i += warpSize) {
        int ky = i / kernel_size;
        int kx = i % kernel_size;
        int in_y = in_y_origin + ky;
        int in_x = in_x_origin + kx;
        float input_val = 0.0f;
        if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
            int input_idx = b * (in_channels * input_h * input_w) +
                            in_ch * (input_h * input_w) +
                            in_y * input_w + in_x;
            input_val = input[input_idx];
        }
        int weight_idx = in_ch * (channels_per_group * kernel_area) +
                         (oc % channels_per_group) * kernel_area +
                         i;
        float wt = weight[weight_idx];
        sum += input_val * wt;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        shared_data[threadIdx.x / warpSize] = sum;
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x / warpSize) {
        float block_sum = shared_data[threadIdx.x];
        if (lane == 0) {
            if(bias != nullptr) {
                block_sum += bias[oc];
            }
            int out_idx = b * (out_channels * output_h * output_w) +
                          oc * (output_h * output_w) +
                          out_h_idx * output_w + out_w_idx;
            output[out_idx] = block_sum;
        }
    }
}

// Forward function to launch the optimized kernel
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    int total_output = batch_size * out_channels * output_h * output_w;
    const int warpSize = 32;
    int total_threads = total_output * warpSize;
    int block_size = 128;
    int grid_size = (total_threads + block_size - 1) / block_size;

    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;
    depthwise_conv2d_optimized_kernel<<<grid_size, block_size, block_size/warpSize*sizeof(float)>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_h,
        input_w,
        out_channels,
        output_h,
        output_w,
        kernel_size,
        stride,
        padding,
        channels_per_group,
        total_output
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution Optimized (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
