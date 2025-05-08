#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__device__ __forceinline__ float compute_conv_sum(
    const float* __restrict__ input,
    const float* __restrict__ sweight,
    int b, int c, int oh, int ow,
    int in_h, int in_w, int kernel_h,
    int stride, int padding, int dilation,
    int channels) {
    
    float sum = 0.f;
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            int in_index = ((b * channels + c) * in_h + ih) * in_w + iw;
            sum += __ldg(&input[in_index]) * sweight[kh];
        }
    }
    return sum;
}

__global__ void depthwise_conv2d_minimal_sync(
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

    const int tile_x = 32;
    const int tile_y = 8;
    
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;
    
    int ow = blockIdx.x * tile_x + threadIdx.x;
    int oh = blockIdx.y * tile_y + threadIdx.y;

    extern __shared__ float sweight[];
    
    // Load weights into shared memory - only done once per block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < kernel_h) {
        sweight[tid] = __ldg(&weight[c * kernel_h + tid]);
    }
    
    // Single synchronization point - only after shared memory is loaded
    __syncthreads();

    // Process output pixels without further synchronization
    if (oh < out_h && ow < out_w) {
        float sum = compute_conv_sum(input, sweight, b, c, oh, ow,
                                   in_h, in_w, kernel_h,
                                   stride, padding, dilation, channels);
        sum += __ldg(&bias[c]);
        output[((b * channels + c) * out_h + oh) * out_w + ow] = sum;
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

    const int tile_x = 32;
    const int tile_y = 8;
    dim3 block(tile_x, tile_y, 1);
    dim3 grid((out_w + tile_x - 1) / tile_x,
              (out_h + tile_y - 1) / tile_y,
              batch * channels);

    int shmem_size = kernel_h * sizeof(float);

    depthwise_conv2d_minimal_sync<<<grid, block, shmem_size>>>(
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
    m.def("forward", &forward, "Minimal sync depthwise 2D convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}