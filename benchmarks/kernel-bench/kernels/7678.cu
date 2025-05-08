#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define TILE_SIZE 8

template <typename scalar_t>
__global__ void conv3d_aligned_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int groups) {

    __shared__ scalar_t shared_input[TILE_SIZE][BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_SIZE_X;
    const int by = blockIdx.y * BLOCK_SIZE_Y;
    const int bz = blockIdx.z;

    const int batch_idx = bz / out_channels;
    const int out_channel = bz % out_channels;
    const int group = out_channel / (out_channels / groups);
    const int channels_per_group = in_channels / groups;

    // Align memory access to 128-bit boundary
    const int aligned_width = (in_width + 3) & ~3;
    const int aligned_out_width = (out_width + 3) & ~3;

    #pragma unroll
    for (int d = 0; d < out_depth; d++) {
        const int h_out = by + ty;
        const int w_out = bx + tx;

        if (h_out < out_height && w_out < out_width) {
            scalar_t sum = 0.0f;

            #pragma unroll
            for (int ic = 0; ic < channels_per_group; ic++) {
                const int in_channel = group * channels_per_group + ic;

                #pragma unroll
                for (int kd = 0; kd < kernel_d; kd++) {
                    const int d_in = d * stride - padding + kd;
                    if (d_in >= 0 && d_in < in_depth) {

                        #pragma unroll
                        for (int kh = 0; kh < kernel_h; kh++) {
                            const int h_in = h_out * stride - padding + kh;
                            if (h_in >= 0 && h_in < in_height) {

                                #pragma unroll
                                for (int kw = 0; kw < kernel_w; kw++) {
                                    const int w_in = w_out * stride - padding + kw;
                                    if (w_in >= 0 && w_in < in_width) {
                                        // Use __ldg for read-only data
                                        const scalar_t input_val = __ldg(&input[
                                            ((batch_idx * in_channels + in_channel) * in_depth + d_in) * 
                                            in_height * aligned_width + h_in * aligned_width + w_in]);
                                        
                                        const scalar_t weight_val = __ldg(&weight[
                                            ((out_channel * channels_per_group + ic) * kernel_d + kd) * 
                                            kernel_h * kernel_w + kh * kernel_w + kw]);

                                        sum += input_val * weight_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (bias != nullptr) {
                sum += __ldg(&bias[out_channel]);
            }

            // Aligned store
            output[
                ((batch_idx * out_channels + out_channel) * out_depth + d) * 
                out_height * aligned_out_width + h_out * aligned_out_width + w_out] = sum;
        }
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {

    auto bias = bias_opt.value_or(at::Tensor());
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "Bias must be a CUDA tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int out_depth = (in_depth + 2 * padding - kernel_d) / stride + 1;
    int out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_w) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(
        (out_width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (out_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        batch_size * out_channels
    );

    conv3d_aligned_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with aligned memory access (CUDA)");
}