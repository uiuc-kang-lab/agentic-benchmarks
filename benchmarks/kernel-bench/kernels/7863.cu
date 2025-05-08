#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16

__global__ void conv2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding) {
    
    extern __shared__ float shared_mem[];
    float* input_tile = shared_mem;
    float* weight_tile = shared_mem + TILE_SIZE * kernel_h * kernel_w;

    const int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= out_channels * batch_size * out_h * out_w) return;

    const int oc = tid / (batch_size * out_h * out_w);
    const int rem = tid % (batch_size * out_h * out_w);
    const int b = rem / (out_h * out_w);
    const int h_out = (rem % (out_h * out_w)) / out_w;
    const int w_out = rem % out_w;

    float sum = 0.0f;

    for (int ic_tile = 0; ic_tile < in_channels; ic_tile += TILE_SIZE) {
        const int ic_limit = min(ic_tile + TILE_SIZE, in_channels);

        // Load input tile into shared memory
        for (int ic = ic_tile; ic < ic_limit; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                const int h_in = h_out * stride + kh - padding;
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int w_in = w_out * stride + kw - padding;
                    const int shmem_idx = (ic - ic_tile) * kernel_h * kernel_w + kh * kernel_w + kw;
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        input_tile[shmem_idx] = __ldg(&input[((b * in_channels + ic) * height + h_in) * width + w_in]);
                    } else {
                        input_tile[shmem_idx] = 0.0f;
                    }
                }
            }
        }

        // Load weight tile into shared memory
        for (int ic = ic_tile; ic < ic_limit; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int shmem_idx = (ic - ic_tile) * kernel_h * kernel_w + kh * kernel_w + kw;
                    weight_tile[shmem_idx] = __ldg(&weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw]);
                }
            }
        }
        __syncthreads();

        // Compute partial sum using shared memory
        for (int ic = 0; ic < ic_limit - ic_tile; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    sum += input_tile[ic * kernel_h * kernel_w + kh * kernel_w + kw] * 
                           weight_tile[ic * kernel_h * kernel_w + kh * kernel_w + kw];
                }
            }
        }
        __syncthreads();
    }

    output[((b * out_channels + oc) * out_h + h_out) * out_w + w_out] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
    CHECK_CUDA(weight); CHECK_CONTIGUOUS(weight);

    if (dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, bias, {stride, stride},
                           {padding, padding}, {dilation, dilation}, groups);
    }

    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto height = x.size(2);
    const auto width = x.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    const auto out_h = (height + 2 * padding - kernel_h) / stride + 1;
    const auto out_w = (width + 2 * padding - kernel_w) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());

    const int threads = 256;
    const int blocks = (out_channels * batch_size * out_h * out_w + threads - 1) / threads;
    const size_t shared_mem_size = 2 * TILE_SIZE * kernel_h * kernel_w * sizeof(float);

    conv2d_shared_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride,
        padding
    );

    if (bias.has_value()) {
        output += bias.value().view({1, -1, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with Shared Memory");
}