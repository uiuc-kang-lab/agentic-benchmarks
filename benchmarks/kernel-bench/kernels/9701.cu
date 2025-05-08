#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Optimized CUDA kernel for depthwise 2D convolution using shared memory reduction
// and warp-level primitives for the final reduction phase.
// Each block computes one output pixel by parallelizing the reduction over the kernel height dimension.
__global__ void depthwise_conv2d_kernel_optimized(
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

    // Each block is responsible for one output pixel
    int out_idx = blockIdx.x;
    int total = batch * channels * out_h * out_w;
    if (out_idx >= total) return;

    // Decode the linear index into 4D coordinates (b, c, oh, ow)
    int tmp = out_idx;
    int ow = tmp % out_w;
    tmp /= out_w;
    int oh = tmp % out_h;
    tmp /= out_h;
    int c  = tmp % channels;
    int b  = tmp / channels;

    float my_sum = 0.0f;
    int tid = threadIdx.x;

    // Each thread in the block computes a partial sum for a subset of kernel elements
    for (int kh = tid; kh < kernel_h; kh += blockDim.x) {
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            int input_index = ((b * channels + c) * in_h + ih) * in_w + iw;
            int weight_index = c * kernel_h + kh; // weight shape: (channels, 1, kernel_h, 1)
            my_sum += input[input_index] * weight[weight_index];
        }
    }

    // Allocate shared memory for intra-block reduction
    extern __shared__ float sdata[];
    sdata[tid] = my_sum;
    __syncthreads();

    // Perform reduction in shared memory; assume blockDim.x is a power of two
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Use warp-level primitives for final reduction among the last 32 threads
    if (tid < 32) {
        float sum_val = sdata[tid];
        // Unroll warp-level reduction using __shfl_down_sync
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        }
        if (tid == 0) {
            sum_val += bias[c];
            output[out_idx] = sum_val;
        }
    }
}

// Forward function to launch the optimized kernel
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

    // Depthwise convolution requires groups equal to number of channels
    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    // Handle bias: if not provided, create a zero bias tensor
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

    // Each block computes one output pixel
    int total = batch * channels * out_h * out_w;
    int blocks = total;
    int threads = 128; // Number of threads per block (should be power of two)
    size_t shared_mem = threads * sizeof(float);

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias_val.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    depthwise_conv2d_kernel_optimized<<<blocks, threads, shared_mem>>>(
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
    m.def("forward", &forward, "Optimized Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
