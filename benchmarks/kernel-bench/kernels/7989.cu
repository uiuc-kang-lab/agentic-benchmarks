#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_warp_reduce_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_height,
    int out_width) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.z / out_channels;
    const int channel = blockIdx.z % out_channels;

    if (b >= batch_size || row >= out_height || col >= out_width || channel >= out_channels) return;

    const int lane_id = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_size = 32;
    const int num_warps = blockDim.x * blockDim.y / warp_size;
    const int warp_id = lane_id / warp_size;

    float sum = 0.0f;
    const int in_row_origin = row * stride - padding;
    const int in_col_origin = col * stride - padding;

    for (int ic = warp_id; ic < in_channels; ic += num_warps) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int in_row = in_row_origin + kh * dilation;
            if (in_row < 0 || in_row >= in_height) continue;
            
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int in_col = in_col_origin + kw * dilation;
                if (in_col < 0 || in_col >= in_width) continue;

                const int input_idx = ((b * in_channels + ic) * in_height + in_row) * in_width + in_col;
                const int weight_idx = ((channel * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
            }
        }
    }

    // Warp-level reduction using warp shuffle
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Only the first thread in each warp writes the result
    if (lane_id % warp_size == 0) {
        const int out_idx = ((b * out_channels + channel) * out_height + row) * out_width + col;
        if (warp_id == 0) {
            // First warp directly writes
            output[out_idx] = sum;
        } else {
            // Other warps use atomic add
            atomicAdd(&output[out_idx], sum);
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
    TORCH_CHECK(groups == 1, "Only groups=1 supported");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    dim3 block(16, 16);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch_size * out_channels
    );

    conv2d_warp_reduce_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        out_height,
        out_width
    );

    if (bias.has_value()) {
        output += bias->view({1, out_channels, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-reduced 2D convolution with shuffle operations");
}
