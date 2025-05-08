#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handles.h>
#include <ATen/cudnn/Descriptors.h>
#include <cudnn.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 4

__global__ void conv3d_shared_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int kernel_d, int kernel_h, int kernel_w,
    int out_depth, int out_height, int out_width,
    int stride, int padding, int dilation, int groups) {

    extern __shared__ float shared_weight[];

    // Calculate the thread indices and dimensions
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // Calculate the starting points for the tile
    int w_out = bx * blockDim.x + tx;
    int h_out = by * blockDim.y + ty;
    int d_out = bz * blockDim.z + tz;

    // Each thread will calculate one element of the output
    float sum = 0.0f;

    int group = bz / (out_channels / groups);
    int in_channels_per_group = in_channels / groups;

    for (int ic = 0; ic < in_channels_per_group; ic++) {
        // Load weights into shared memory
        if (tx < kernel_w && ty < kernel_h && tz < kernel_d) {
            int weight_index = ((bz * in_channels_per_group + ic) * kernel_d + tz) * kernel_h * kernel_w + ty * kernel_w + tx;
            shared_weight[tz * kernel_h * kernel_w + ty * kernel_w + tx] = weight[weight_index];
        }
        __syncthreads();

        // Compute aggregation
        for (int kd = 0; kd < kernel_d; kd++) {
            int d_in = d_out * stride - padding + kd * dilation;
            if (d_in < 0 || d_in >= in_depth) continue;

            for (int kh = 0; kh < kernel_h; kh++) {
                int h_in = h_out * stride - padding + kh * dilation;
                if (h_in < 0 || h_in >= in_height) continue;

                for (int kw = 0; kw < kernel_w; kw++) {
                    int w_in = w_out * stride - padding + kw * dilation;
                    if (w_in < 0 || w_in >= in_width) continue;

                    int input_index = ((blockIdx.z * in_channels + ic) * in_depth + d_in) * in_height * in_width +
                                      h_in * in_width + w_in;
                    sum += input[input_index] * shared_weight[kd * kernel_h * kernel_w + kh * kernel_w + kw];
                }
            }
        }
        __syncthreads();
    }

    if (bias != nullptr) {
        sum += bias[blockIdx.z];
    }

    // Write the output
    int output_index = (((blockIdx.z * blockDim.z + tz) * out_height + h_out) * out_width + w_out);
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

    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 numBlocks(
        (out_width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (out_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        (out_depth + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z
    );

    size_t sharedMemSize = kernel_d * kernel_h * kernel_w * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_shared_kernel", ([&] {
        conv3d_shared_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
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
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward using shared memory (CUDA)");
}
