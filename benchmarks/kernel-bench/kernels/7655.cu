#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Kernel using shared memory to optimize memory access patterns
__global__ void conv3d_shared_memory_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int out_depth,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups) {

    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int d_out = blockIdx.z * blockDim.z + threadIdx.z;

    if (w_out >= out_width || h_out >= out_height || d_out >= out_depth) return;

    int c_out = blockIdx.w;
    int b = blockIdx.v;

    float sum = 0.0f;

    int out_channels_per_group = out_channels / groups;
    int group = c_out / out_channels_per_group;
    int in_channels_per_group = in_channels / groups;

    for (int ic = 0; ic < in_channels_per_group; ic++) {
        int in_c = group * in_channels_per_group + ic;

        for (int kd = 0; kd < kernel_d; kd++) {
            int d_in = d_out * stride - padding + kd * dilation;
            if (d_in < 0 || d_in >= in_depth) continue;

            for (int kh = 0; kh < kernel_h; kh++) {
                int h_in = h_out * stride - padding + kh * dilation;
                if (h_in < 0 || h_in >= in_height) continue;

                for (int kw = 0; kw < kernel_w; kw++) {
                    int w_in = w_out * stride - padding + kw * dilation;
                    if (w_in < 0 || w_in >= in_width) continue;

                    int input_index = ((b * in_channels + in_c) * in_depth + d_in) * in_height * in_width +
                                      h_in * in_width + w_in;
                    int weight_index = (((c_out * in_channels_per_group) + ic) * kernel_d + kd) *
                                       kernel_h * kernel_w + kh * kernel_w + kw;

                    shared_input[tid] = input[input_index];
                    shared_weight[tid] = weight[weight_index];

                    __syncthreads();

                    sum += shared_input[tid] * shared_weight[tid];

                    __syncthreads();
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    int output_index = ((b * out_channels + c_out) * out_depth + d_out) * out_height * out_width +
                       h_out * out_width + w_out;
    output[output_index] = sum;
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto bias = bias_opt.value_or(at::Tensor());
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(
        (out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (out_depth + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    size_t shared_memory_size = 2 * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

    conv3d_shared_memory_kernel<<<numBlocks, threadsPerBlock, shared_memory_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation, groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with shared memory optimization (CUDA)");
}
