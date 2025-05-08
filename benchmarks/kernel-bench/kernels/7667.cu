#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handles.h>
#include <ATen/cudnn/Descriptors.h>
#include <cudnn.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 4

// Helper function to map at::ScalarType to cudnnDataType_t
cudnnDataType_t getCudnnDataType(at::ScalarType type) {
    switch (type) {
        case at::ScalarType::Float:
            return CUDNN_DATA_FLOAT;
        case at::ScalarType::Double:
            return CUDNN_DATA_DOUBLE;
        case at::ScalarType::Half:
            return CUDNN_DATA_HALF;
        default:
            TORCH_CHECK(false, "Unsupported data type for cuDNN");
    }
}

__global__ void conv3d_shared_memory_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int kernel_d, int kernel_h, int kernel_w,
    int out_depth, int out_height, int out_width,
    int stride, int padding, int dilation, int groups) {

    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int w_out = blockIdx.x * blockDim.x + tid_x;
    int h_out = blockIdx.y * blockDim.y + tid_y;
    int d_out = blockIdx.z * blockDim.z + tid_z;

    if (w_out >= out_width || h_out >= out_height || d_out >= out_depth) return;

    int c_out = blockIdx.z;
    int b = blockIdx.y;

    float sum = 0.0f;

    int group = c_out / (out_channels / groups);
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
                    int weight_index = (((c_out * in_channels_per_group) + ic) * kernel_d + kd) * kernel_h * kernel_w +
                                       kh * kernel_w + kw;

                    shared_input[tid_z * BLOCK_SIZE_Y * BLOCK_SIZE_X + tid_y * BLOCK_SIZE_X + tid_x] = input[input_index];
                    shared_weight[tid_z * BLOCK_SIZE_Y * BLOCK_SIZE_X + tid_y * BLOCK_SIZE_X + tid_x] = weight[weight_index];

                    __syncthreads();

                    sum += shared_input[tid_z * BLOCK_SIZE_Y * BLOCK_SIZE_X + tid_y * BLOCK_SIZE_X + tid_x] *
                           shared_weight[tid_z * BLOCK_SIZE_Y * BLOCK_SIZE_X + tid_y * BLOCK_SIZE_X + tid_x];

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
    // Ensure inputs are on CUDA
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "Bias must be a CUDA tensor");

    // Get input dimensions
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_depth = input.size(2);
    int64_t in_height = input.size(3);
    int64_t in_width = input.size(4);

    // Get weight dimensions
    int64_t out_channels = weight.size(0);
    int64_t kernel_d = weight.size(2);
    int64_t kernel_h = weight.size(3);
    int64_t kernel_w = weight.size(4);

    // Calculate output dimensions
    int64_t out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int64_t out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    // Prepare output tensor
    auto options = input.options();
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, options);

    // Launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 numBlocks(
        (out_width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (out_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        (out_depth + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z
    );

    size_t shared_memory_size = (BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z) * sizeof(float) * 2;

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
    m.def("forward", &forward, "3D convolution forward using shared memory optimization (CUDA)");
}