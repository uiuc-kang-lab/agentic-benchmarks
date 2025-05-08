#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv_transpose2d_kernel(
float* output,
const float* input,
const float* weight,
const float* bias,
int batch_size,
int in_channels,
int out_channels,
int in_height,
int in_width,
int kernel_h,
int kernel_w,
int stride_h,
int stride_w,
int pad_h,
int pad_w) {

    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + blockDim.x * blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Load input and weight data to shared memory
    if (tx < in_width && ty < in_height) {
        shared_input[ty * in_width + tx] = input[by * in_height * in_width + ty * in_width + tx];
    }
    if (tx < kernel_w && ty < kernel_h) {
        shared_weight[ty * kernel_w + tx] = weight[bx * kernel_h * kernel_w + ty * kernel_w + tx];
    }
    __syncthreads();

    int out_h = (in_height - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_w = (in_width - 1) * stride_w - 2 * pad_w + kernel_w;

    int out_y = blockDim.y * blockIdx.y + threadIdx.y;
    int out_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (out_y < out_h && out_x < out_w) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int ky = 0; ky < kernel_h; ky++) {
            #pragma unroll
            for (int kx = 0; kx < kernel_w; kx++) {
                int in_y = (out_y + pad_h - ky) / stride_h;
                int in_x = (out_x + pad_w - kx) / stride_w;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    sum += shared_input[in_y * in_width + in_x] * 
                           shared_weight[ky * kernel_w + kx];
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[bx];
        }

        output[by * out_h * out_w + out_y * out_w + out_x] = sum;
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

    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    
    const dim3 threads(16, 16);
    const dim3 blocks(
        (in_width + threads.x - 1) / threads.x,
        (in_height + threads.y - 1) / threads.y
    );

    const int shared_mem_size = 
        (in_height * in_width + kernel_h * kernel_w) * sizeof(float);

    auto output = at::zeros({
        batch_size,
        weight.size(1),
        (in_height - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0],
        (in_width - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    }, x.options());

    conv_transpose2d_kernel<<<blocks, threads, shared_mem_size>>>(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        batch_size,
        in_channels,
        weight.size(1),
        in_height,
        in_width,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}