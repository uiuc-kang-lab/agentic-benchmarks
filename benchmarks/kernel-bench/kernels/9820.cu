#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel for depthwise 2D convolution with branchless boundary handling
// using clamped indices and computed valid masks to minimize warp divergence.
__global__ void depthwise_conv2d_kernel(
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

    // Shared memory for input tile and weights
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = &shared_mem[blockDim.x + 2*padding];

    // Decode output coordinates
    int ow = threadIdx.x + blockIdx.x * blockDim.x;
    int oh = blockIdx.y % out_h;
    int c  = blockIdx.y / out_h;
    int b  = blockIdx.z;

    // Load weights into shared memory (one per thread)
    if (threadIdx.x < kernel_h && c < channels) {
        shared_weight[threadIdx.x] = weight[c * kernel_h + threadIdx.x];
    }
    
    if (ow < out_w && c < channels && b < batch) {
        float sum = 0.0f;

        // Pre-compute offsets for input indexing
        int input_batch_offset = b * channels * in_h * in_w;
        int input_channel_offset = c * in_h * in_w;

        // Load input row into shared memory with padding
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            int safe_ih = min(max(ih, 0), in_h - 1);
            float valid_y = (ih >= 0 && ih < in_h) ? 1.0f : 0.0f;

            // Load input elements into shared memory
            int base_iw = ow * stride - padding;
            int safe_iw = min(max(base_iw, 0), in_w - 1);
            float valid_x = (base_iw >= 0 && base_iw < in_w) ? 1.0f : 0.0f;

            int input_idx = input_batch_offset + input_channel_offset + safe_ih * in_w + safe_iw;
            float in_val = __ldg(&input[input_idx]);
            
            // Compute convolution using shared memory
            sum += valid_x * valid_y * in_val * shared_weight[kh];
        }
        
        sum += __ldg(&bias[c]);
        int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[output_idx] = sum;
    }

    __syncthreads();
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

    // Ensure contiguous tensors
    x = x.contiguous();
    weight = weight.contiguous();

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);  // weight shape: (channels, 1, kernel_h, 1)

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

    // Launch parameters: using 256 threads per block for the width dimension
    int block_size = 256;
    dim3 threads(block_size, 1, 1);
    dim3 blocks(
        (out_w + block_size - 1) / block_size,
        channels * out_h,
        batch
    );

    depthwise_conv2d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
