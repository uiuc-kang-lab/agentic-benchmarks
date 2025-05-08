#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void depthwise_conv2d_coalesced(
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

    // Use 32x8 thread configuration for better memory coalescing
    const int BLOCK_WIDTH = 32;  // Match warp size for coalesced access
    const int BLOCK_HEIGHT = 8;

    // Calculate batch and channel index
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;

    // Calculate base output coordinates
    int ow_base = blockIdx.x * BLOCK_WIDTH;
    int oh_base = blockIdx.y * BLOCK_HEIGHT;

    // Thread indices within the block
    int tx = threadIdx.x;  // Range: 0-31
    int ty = threadIdx.y;  // Range: 0-7

    // Calculate actual output coordinates
    int ow = ow_base + tx;
    int oh = oh_base + ty;

    // Cache weights in shared memory
    extern __shared__ float sweight[];
    int tid = ty * BLOCK_WIDTH + tx;
    if (tid < kernel_h) {
        sweight[tid] = __ldg(&weight[c * kernel_h + tid]);
    }
    __syncthreads();

    if (oh < out_h && ow < out_w) {
        float sum = 0.0f;
        
        // Process kernel height with unrolling for better instruction throughput
        #pragma unroll
        for (int kh = 0; kh < kernel_h; kh++) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding;

            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                // Calculate input index ensuring coalesced access within warps
                int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                float in_val = __ldg(&input[in_idx]);
                sum += in_val * sweight[kh];
            }
        }

        // Add bias
        sum += __ldg(&bias[c]);

        // Calculate output index ensuring coalesced writes
        int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
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

    // Configure block and grid dimensions for coalesced memory access
    const int BLOCK_WIDTH = 32;  // Match warp size
    const int BLOCK_HEIGHT = 8;
    
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(
        (out_w + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
        (out_h + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT,
        batch * channels
    );

    int shmem_size = kernel_h * sizeof(float);

    depthwise_conv2d_coalesced<<<grid, block, shmem_size>>>(
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
    m.def("forward", &forward, "Coalesced Memory Optimized Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}