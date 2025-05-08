#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

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

    // Determine batch and channel indices
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;

    // Use 32x8 thread configuration for better memory coalescing
    const int tile_x = 32;
    const int tile_y = 8;

    // Calculate output positions
    int ow = blockIdx.x * tile_x + threadIdx.x;
    int oh = blockIdx.y * tile_y + threadIdx.y;

    // Load weights into shared memory - single synchronization point
    extern __shared__ float sweight[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < kernel_h) {
        sweight[tid] = __ldg(&weight[c * kernel_h + tid]);
    }
    __syncthreads();  // Single sync point after weight loading

    // Process output elements
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
        
        // Add bias and write output
        sum += __ldg(&bias[c]);
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

    // Configure kernel launch parameters
    const int tile_x = 32;
    const int tile_y = 8;
    dim3 threads(tile_x, tile_y);
    dim3 blocks((out_w + tile_x - 1) / tile_x,
                (out_h + tile_y - 1) / tile_y,
                batch * channels);

    // Launch kernel with shared memory for weights
    int shmem_size = kernel_h * sizeof(float);
    
    depthwise_conv2d_minimal_sync<<<blocks, threads, shmem_size>>>(
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
    m.def("forward", &forward, "Minimal sync depthwise conv2d forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}