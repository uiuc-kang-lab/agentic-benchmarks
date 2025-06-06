#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Device function to load weights into shared memory
__device__ void load_weights_to_shared(const float* __restrict__ weight, float* sweight, int c, int kernel_h) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < kernel_h) {
        sweight[tid] = __ldg(&weight[c * kernel_h + tid]);
    }
    __syncthreads();
}

// Device function to compute the convolution sum for one element
__device__ float compute_element_sum(
    const float* __restrict__ input,
    const float* sweight,
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
            float in_val = __ldg(&input[in_index]);
            sum += in_val * sweight[kh];
        }
    }
    return sum;
}

// Main kernel function
__global__ void depthwise_conv2d_modular_kernel(
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

    // Tile dimensions for output per block
    const int tile_x = 16;
    const int tile_y = 16;

    // Compute starting indices for this tile
    int ow_start = blockIdx.x * tile_x;
    int oh_start = blockIdx.y * tile_y;

    // Each thread computes one output element in the tile
    int ow = ow_start + threadIdx.x;
    int oh = oh_start + threadIdx.y;

    // Allocate shared memory to cache the kernel weights for this channel
    extern __shared__ float sweight[]; // size: kernel_h * sizeof(float)
    load_weights_to_shared(weight, sweight, c, kernel_h);

    if (oh < out_h && ow < out_w) {
        float sum = compute_element_sum(
            input, sweight, b, c, oh, ow,
            in_h, in_w, kernel_h,
            stride, padding, dilation, channels) + __ldg(&bias[c]);
        int out_index = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[out_index] = sum;
    }
}

// Forward function with modular kernel refactoring
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Ensure inputs are contiguous
    x = x.contiguous();
    weight = weight.contiguous();

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);  // weight shape: (channels, 1, kernel_h, 1)

    // Depthwise convolution requires groups == channels
    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    // Handle bias: if bias is not provided, use a zeros tensor
    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }

    // Compute output dimensions
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;

    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    // Define tile dimensions
    const int tile_x = 16;
    const int tile_y = 16;
    dim3 block(tile_x, tile_y, 1);
    dim3 grid((out_w + tile_x - 1) / tile_x, (out_h + tile_y - 1) / tile_y, batch * channels);

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias_val.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Allocate shared memory size: kernel_h elements
    int shmem_size = kernel_h * sizeof(float);

    depthwise_conv2d_modular_kernel<<<grid, block, shmem_size>>>(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
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
    m.def("forward", &forward, "Modular Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}