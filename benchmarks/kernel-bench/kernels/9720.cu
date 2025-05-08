#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void depthwise_conv2d_optimized_mapping(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int stride,
    int padding,
    int dilation) {

    // Calculate global thread index
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    // Calculate batch and channel from block index
    const int bc = blockIdx.y;
    const int b = bc / channels;
    const int c = bc % channels;

    // Cache weights in shared memory
    extern __shared__ float sweight[];
    if (threadIdx.x < kernel_h) {
        sweight[threadIdx.x] = __ldg(&weight[c * kernel_h + threadIdx.x]);
    }
    __syncthreads();

    // Each thread processes multiple output elements
    for (int idx = thread_idx; idx < out_h * out_w; idx += total_threads) {
        const int ow = idx % out_w;
        const int oh = idx / out_w;

        float sum = 0.0f;
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int ih = oh * stride - padding + kh * dilation;
            const int iw = ow * stride - padding;
            
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                const int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                sum += __ldg(&input[in_idx]) * sweight[kh];
            }
        }
        
        sum += __ldg(&bias[c]);
        const int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[out_idx] = sum;
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    x = x.contiguous();
    weight = weight.contiguous();

    const int batch = x.size(0);
    const int channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_h = weight.size(2);

    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }

    const int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - 1) / stride + 1;

    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    // Optimize thread block configuration
    const int thread_block_size = 256;
    const int num_blocks_x = (out_h * out_w + thread_block_size - 1) / thread_block_size;
    const int num_blocks_y = batch * channels;

    dim3 grid(num_blocks_x, num_blocks_y);
    dim3 block(thread_block_size);

    const int shared_mem_size = kernel_h * sizeof(float);

    depthwise_conv2d_optimized_mapping<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_val.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        stride,
        padding,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution forward with optimized thread mapping (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}