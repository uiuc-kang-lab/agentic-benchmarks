#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(const float* __restrict__ input, const float* __restrict__ weight, float* __restrict__ output,
                              int input_height, int input_width, int kernel_size, int stride, int padding, int output_height, int output_width) {
    extern __shared__ float shared_mem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int output_x = bx * blockDim.x + tx;
    int output_y = by * blockDim.y + ty;

    if (output_x < output_width && output_y < output_height) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int input_x = output_x * stride + kx - padding;
                int input_y = output_y * stride + ky - padding;
                if (input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height) {
                    sum += input[input_y * input_width + input_x] * weight[ky * kernel_size + kx];
                }
            }
        }
        output[output_y * output_width + output_x] = sum;
        __syncthreads();

        // Warp-level reduction
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (tx % warpSize == 0) {
            output[output_y * output_width + output_x] = sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    auto output = torch::empty({x.size(0), weight.size(0), (x.size(2) - weight.size(2) + 2 * padding) / stride + 1, (x.size(3) - weight.size(3) + 2 * padding) / stride + 1}, x.options());

    int threads = 32;
    dim3 block(threads, threads);
    dim3 grid((output.size(3) + block.x - 1) / block.x, (output.size(2) + block.y - 1) / block.y);
    size_t shared_mem_size = threads * threads * sizeof(float);

    conv2d_kernel<<<grid, block, shared_mem_size>>>(x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
                                                    x.size(2), x.size(3), weight.size(2), stride, padding, output.size(2), output.size(3));

    if (bias.has_value()) {
        output += bias.value().view({1, -1, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with shared memory and warp optimization");
}