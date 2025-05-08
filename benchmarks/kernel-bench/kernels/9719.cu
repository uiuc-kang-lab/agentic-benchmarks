#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <algorithm>

// Kernel that supports pipelined execution by processing a sub-batch defined by the batch_offset.
__global__ void depthwise_conv2d_pipeline_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_offset,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int stride,
    int padding,
    int dilation) {

    // Each block in the grid.z dimension corresponds to one (batch, channel) pair within the sub-batch
    int bc = blockIdx.z;           // local index for batch and channel
    int b_local = bc / channels;   // local batch index
    int c = bc % channels;         // channel index
    int b = batch_offset + b_local; // global batch index

    // Tile dimensions
    const int tile_x = 16;
    const int tile_y = 16;

    int ow_start = blockIdx.x * tile_x;
    int oh_start = blockIdx.y * tile_y;
    int ow = ow_start + threadIdx.x;
    int oh = oh_start + threadIdx.y;

    // Allocate shared memory to cache the kernel weights for the current channel
    extern __shared__ float sweight[]; // size: kernel_h * sizeof(float)
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < kernel_h) {
        sweight[tid] = __ldg(&weight[c * kernel_h + tid]);
    }
    __syncthreads();

    if (oh < out_h && ow < out_w) {
        float sum = 0.f;
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding; // kernel width is 1
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int in_index = ((b * channels + c) * in_h + ih) * in_w + iw;
                sum += __ldg(&input[in_index]) * sweight[kh];
            }
        }
        sum += __ldg(&bias[c]);
        int out_index = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[out_index] = sum;
    }
}

// Forward function implementing pipelined multi-stream processing to overlap computation with memory transfers
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Ensure the inputs are contiguous
    x = x.contiguous();
    weight = weight.contiguous();

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);  // weight shape: (channels, 1, kernel_h, 1)

    // Check depthwise condition
    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    // Handle bias: if not provided, create a zeros tensor
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

    // Overlap computation with memory transfers by splitting the batch across multiple CUDA streams
    int num_streams = std::min(batch, 4); // Use up to 4 streams
    int batch_per_stream = (batch + num_streams - 1) / num_streams;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int tile_x = 16;
    const int tile_y = 16;
    dim3 block(tile_x, tile_y, 1);
    int grid_x = (out_w + tile_x - 1) / tile_x;
    int grid_y = (out_h + tile_y - 1) / tile_y;

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias_val.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    int shmem_size = kernel_h * sizeof(float);

    // Launch kernels for each sub-batch asynchronously
    for (int s = 0; s < num_streams; ++s) {
        int batch_offset = s * batch_per_stream;
        int local_batch = std::min(batch_per_stream, batch - batch_offset);
        if (local_batch <= 0) break;
        dim3 grid(grid_x, grid_y, local_batch * channels);
        depthwise_conv2d_pipeline_kernel<<<grid, block, shmem_size, streams[s]>>>(
            x_ptr,
            weight_ptr,
            bias_ptr,
            output_ptr,
            batch_offset,
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
    }

    // Synchronize and destroy streams
    for (int s = 0; s < num_streams; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution forward with pipelined streams (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
