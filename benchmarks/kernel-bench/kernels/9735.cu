#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void depthwise_conv2d_strided(
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

    // Determine batch and channel based on blockIdx.z
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;

    // Load weights into shared memory
    extern __shared__ float sweight[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < kernel_h) {
        sweight[tid] = __ldg(&weight[c * kernel_h + tid]);
    }
    __syncthreads();

    // Each thread processes multiple elements with stride
    const int elements_per_thread = 4;
    const int thread_stride = blockDim.x * blockDim.y;
    
    // Calculate base output position
    int base_oh = blockIdx.y * blockDim.y + threadIdx.y;
    int base_ow = blockIdx.x * blockDim.x + threadIdx.x;

    // Process multiple elements per thread using stride pattern
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int oh = base_oh;
        int ow = base_ow + i * thread_stride;

        if (oh < out_h && ow < out_w) {
            float sum = 0.0f;
            
            #pragma unroll
            for (int kh = 0; kh < kernel_h; ++kh) {
                int ih = oh * stride - padding + kh * dilation;
                int iw = ow * stride - padding;

                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                    sum += __ldg(&input[in_idx]) * sweight[kh];
                }
            }

            sum += __ldg(&bias[c]);
            int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
            output[out_idx] = sum;
        }
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

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);

    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }

    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;

    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    // Adjust block dimensions for strided access
    const int block_x = 16;
    const int block_y = 16;
    dim3 block(block_x, block_y);
    
    // Adjust grid dimensions considering elements_per_thread
    const int elements_per_thread = 4;
    dim3 grid(
        (out_w + block_x * elements_per_thread - 1) / (block_x * elements_per_thread),
        (out_h + block_y - 1) / block_y,
        batch * channels
    );

    int shmem_size = kernel_h * sizeof(float);

    depthwise_conv2d_strided<<<grid, block, shmem_size>>>(
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
    m.def("forward", &forward, "Strided Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}