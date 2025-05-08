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

    // Use 32-thread warps for coalesced memory access
    const int warp_size = 32;
    const int warps_per_block = blockDim.x / warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int lane_id = threadIdx.x % warp_size;
    
    // Calculate global indices
    const int batch_idx = blockIdx.z / channels;
    const int channel_idx = blockIdx.z % channels;
    
    // Calculate output coordinates
    const int out_x = blockIdx.x * warp_size + lane_id;
    const int out_y = blockIdx.y * warps_per_block + warp_id;

    // Load weights into shared memory
    extern __shared__ float shared_weights[];
    if (threadIdx.x < kernel_h) {
        shared_weights[threadIdx.x] = __ldg(&weight[channel_idx * kernel_h + threadIdx.x]);
    }
    __syncthreads();

    if (out_x < out_w && out_y < out_h) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int k = 0; k < kernel_h; k++) {
            const int in_y = out_y * stride - padding + k * dilation;
            const int in_x = out_x * stride - padding;
            
            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                const int in_idx = ((batch_idx * channels + channel_idx) * in_h + in_y) * in_w + in_x;
                sum += __ldg(&input[in_idx]) * shared_weights[k];
            }
        }
        
        // Add bias
        sum += __ldg(&bias[channel_idx]);
        
        // Write output with coalesced access
        const int out_idx = ((batch_idx * channels + channel_idx) * out_h + out_y) * out_w + out_x;
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

    // Configure block and grid dimensions for coalesced access
    const int warp_size = 32;
    const int warps_per_block = 8;
    const int threads_per_block = warp_size * warps_per_block;
    
    dim3 block(threads_per_block);
    dim3 grid(
        (out_w + warp_size - 1) / warp_size,
        (out_h + warps_per_block - 1) / warps_per_block,
        batch * channels
    );

    const int shared_mem_size = kernel_h * sizeof(float);

    depthwise_conv2d_coalesced<<<grid, block, shared_mem_size>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution forward with coalesced memory access (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}