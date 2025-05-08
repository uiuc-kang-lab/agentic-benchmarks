#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile size for output block (TILE_SIZE x TILE_SIZE threads per block)
#define TILE_SIZE 16

// CUDA Kernel: Depthwise Convolution using Shared Memory and zero-padding preload
// to eliminate divergent branches in the inner loop.
// Assumes asymmetric kernel with kernel width = 1.

__global__ void depthwise_conv2d_shared_nodiv_kernel(
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

    // Determine output tile starting indices
    int out_row_start = blockIdx.y * TILE_SIZE;
    int out_col_start = blockIdx.x * TILE_SIZE;

    // Determine batch and channel from grid's z dimension
    int channel = blockIdx.z % channels;
    int batch_idx = blockIdx.z / channels;

    // Compute corresponding starting indices in input (with padding offset)
    int in_row_start = out_row_start * stride - padding;
    int in_col_start = out_col_start * stride - padding;

    // Shared memory tile dimensions:
    // We need to load all input elements required to compute a TILE_SIZE x TILE_SIZE output tile.
    // For each output row, we require an input span of stride, and an extra (kernel_h-1)*dilation rows for the kernel.
    int sm_rows = TILE_SIZE * stride + (kernel_h - 1) * dilation;
    int sm_cols = TILE_SIZE * stride; // kernel width is 1, so horizontal span is TILE_SIZE*stride

    // Declare shared memory; size provided dynamically
    extern __shared__ float smem[];

    // Load shared memory tile from global memory.
    // Use all threads in block to cooperatively load the tile.
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    int smem_size = sm_rows * sm_cols;
    
    for (int idx = thread_id; idx < smem_size; idx += total_threads) {
        int r = idx / sm_cols;
        int c = idx % sm_cols;
        int global_r = in_row_start + r;
        int global_c = in_col_start + c;
        float val = 0.0f;
        // Avoid divergent branching by using a uniform load of 0 for out-of-bound indices
        if (global_r >= 0 && global_r < in_h && global_c >= 0 && global_c < in_w) {
            int input_idx = ((batch_idx * channels + channel) * in_h + global_r) * in_w + global_c;
            val = input[input_idx];
        }
        smem[idx] = val;
    }
    __syncthreads();

    // Each thread computes one output element of the tile
    int local_y = threadIdx.y; // tile local row index
    int local_x = threadIdx.x; // tile local col index
    int out_y = out_row_start + local_y;
    int out_x = out_col_start + local_x;

    if (out_y < out_h && out_x < out_w) {
        float sum = 0.0f;
        // The corresponding position in shared memory for this output pixel
        // is at (local_y * stride, local_x * stride) relative to smem tile
        int base_row = local_y * stride;
        int base_col = local_x * stride;

        // Unrolled loop over kernel height with uniform control flow
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            int sm_r = base_row + kh * dilation;
            int sm_c = base_col; // kernel width is 1
            int sm_idx = sm_r * sm_cols + sm_c;
            sum += weight[channel * kernel_h + kh] * smem[sm_idx];
        }
        sum += bias[channel];

        int output_idx = ((batch_idx * channels + channel) * out_h + out_y) * out_w + out_x;
        output[output_idx] = sum;
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
    int kernel_h = weight.size(2);  // Assumes kernel shape: (channels, 1, kernel_h, 1)

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

    // Grid dimensions: each block computes a TILE_SIZE x TILE_SIZE output tile for one channel of one batch element
    dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
              (out_h + TILE_SIZE - 1) / TILE_SIZE,
              batch * channels);
    dim3 block(TILE_SIZE, TILE_SIZE);

    // Compute shared memory size needed: sm_rows x sm_cols floats
    int sm_rows = TILE_SIZE * stride + (kernel_h - 1) * dilation;
    int sm_cols = TILE_SIZE * stride;
    size_t shared_mem_size = sm_rows * sm_cols * sizeof(float);
    
    depthwise_conv2d_shared_nodiv_kernel<<<grid, block, shared_mem_size>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution forward (CUDA) - Shared Memory, No Divergence",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
