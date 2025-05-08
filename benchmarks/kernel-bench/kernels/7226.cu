#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_optimized_kernel(
    const float * __restrict__ input,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float * __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_size,
    int out_h,
    int out_w,
    int stride,
    int padding) {

    extern __shared__ float shared_mem[];
    float* sh_weight = shared_mem;
    float* sh_input = &shared_mem[in_channels * kernel_size * kernel_size];

    int oc = blockIdx.z % out_channels;
    int n  = blockIdx.z / out_channels;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int idx = tid; idx < in_channels * kernel_size * kernel_size; idx += blockDim.x * blockDim.y) {
        sh_weight[idx] = weight[oc * (in_channels * kernel_size * kernel_size) + idx];
    }

    const int TILE_SIZE = 16;
    float sum = 0.0f;

    if (out_row < out_h && out_col < out_w) {
        for (int ic = 0; ic < in_channels; ic++) {
            for (int i = 0; i < kernel_size; i++) {
                int in_row = out_row * stride - padding + i;
                if (in_row >= 0 && in_row < in_h) {
                    for (int j = 0; j < kernel_size; j++) {
                        int in_col = out_col * stride - padding + j;
                        if (in_col >= 0 && in_col < in_w) {
                            sh_input[i * kernel_size + j] = 
                                input[n * (in_channels * in_h * in_w) + 
                                      ic * (in_h * in_w) + 
                                      in_row * in_w + in_col];
                        } else {
                            sh_input[i * kernel_size + j] = 0.0f;
                        }
                    }
                } else {
                    for (int j = 0; j < kernel_size; j++) {
                        sh_input[i * kernel_size + j] = 0.0f;
                    }
                }
            }
            __syncthreads();

            for (int ki = 0; ki < kernel_size; ki++) {
                for (int kj = 0; kj < kernel_size; kj++) {
                    sum += sh_input[ki * kernel_size + kj] * 
                           sh_weight[ic * kernel_size * kernel_size + ki * kernel_size + kj];
                }
            }
            __syncthreads();
        }

        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[n * (out_channels * out_h * out_w) + 
               oc * (out_h * out_w) + 
               out_row * out_w + out_col] = sum;
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

    if (groups == 1 && dilation == 1 && x.dim() == 4 && weight.dim() == 4) {
        auto batch_size = x.size(0);
        auto in_channels = x.size(1);
        auto in_h = x.size(2);
        auto in_w = x.size(3);
        auto out_channels = weight.size(0);
        auto kernel_size = weight.size(2);
        auto out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
        auto out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

        auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

        dim3 block(16, 16, 1);
        dim3 grid((out_w + block.x - 1) / block.x,
                 (out_h + block.y - 1) / block.y,
                 batch_size * out_channels);

        size_t shared_mem_size = (in_channels * kernel_size * kernel_size + kernel_size * kernel_size) * sizeof(float);

        conv2d_optimized_kernel<<<grid, block, shared_mem_size>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, in_h, in_w,
            out_channels, kernel_size, out_h, out_w,
            stride, padding
        );

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

        return output;
    }

    return torch::conv2d(x, weight, 
                        bias.has_value() ? bias.value() : torch::Tensor(),
                        {stride, stride}, 
                        {padding, padding}, 
                        {dilation, dilation}, 
                        groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive CUDA convolution forward");
}