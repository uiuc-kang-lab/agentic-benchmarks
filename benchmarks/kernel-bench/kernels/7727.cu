#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void conv3d_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int dilation,
    int groups,
    int out_depth,
    int out_height,
    int out_width
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    if (d >= out_depth || h >= out_height || w >= out_width) return;

    int n = blockIdx.z / out_channels;
    int c_out = blockIdx.z % out_channels;

    T value = 0.0;
    int c_in_start = (c_out / (out_channels / groups)) * (in_channels / groups);
    int c_in_end = c_in_start + (in_channels / groups);

    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            int input_d = d * stride - padding + kd * dilation;
            if (input_d < 0 || input_d >= in_depth) continue;
            for (int kh = 0; kh < kernel_h; ++kh) {
                int input_h = h * stride - padding + kh * dilation;
                if (input_h < 0 || input_h >= in_height) continue;
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int input_w = w * stride - padding + kw * dilation;
                    if (input_w < 0 || input_w >= in_width) continue;

                    T input_val = input[n * in_channels * in_depth * in_height * in_width +
                                      c_in * in_depth * in_height * in_width +
                                      input_d * in_height * in_width +
                                      input_h * in_width +
                                      input_w];

                    T weight_val = weight[c_out * (in_channels / groups) * kernel_d * kernel_h * kernel_w +
                                        (c_in - c_in_start) * kernel_d * kernel_h * kernel_w +
                                        kd * kernel_h * kernel_w +
                                        kh * kernel_w +
                                        kw];

                    value += input_val * weight_val;
                }
            }
        }
    }

    if (bias) {
        value += bias[c_out];
    }

    output[n * out_channels * out_depth * out_height * out_width +
          c_out * out_depth * out_height * out_width +
          d * out_height * out_width +
          h * out_width +
          w] = value;
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
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "Bias must be a CUDA tensor");

    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_depth = input.size(2);
    int64_t in_height = input.size(3);
    int64_t in_width = input.size(4);

    int64_t out_channels = weight.size(0);
    int64_t kernel_d = weight.size(2);
    int64_t kernel_h = weight.size(3);
    int64_t kernel_w = weight.size(4);

    int64_t out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int64_t out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    dim3 block(4, 8, 8);  // 256 threads per block
    dim3 grid(
        (out_depth + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        (batch_size * out_channels + block.z - 1) / block.z
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_forward", [&] {
        conv3d_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_depth,
            in_height,
            in_width,
            out_channels,
            kernel_d,
            kernel_h,
            kernel_w,
            stride,
            padding,
            dilation,
            groups,
            out_depth,
            out_height,
            out_width
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D convolution forward (CUDA)");
}