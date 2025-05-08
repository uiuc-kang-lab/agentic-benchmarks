#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Define tile size for input channels to be loaded into shared memory
#define TILE_IC 16

// CUDA kernel that leverages shared memory for weight tiles
__global__ void conv_transpose2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    // Allocate dynamic shared memory for a tile of weights
    extern __shared__ float shared_weight[];

    // Each block is assigned to a specific batch element and output channel
    int block_id = blockIdx.z;
    int out_c = block_id % out_channels;
    int b = block_id / out_channels;

    // Compute the output pixel coordinate for this thread
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_y >= out_height || out_x >= out_width) return;

    float sum = 0.0f;

    // Loop over input channels in tiles
    for (int ic_tile = 0; ic_tile < in_channels; ic_tile += TILE_IC) {
        int current_tile = (TILE_IC < (in_channels - ic_tile) ? TILE_IC : (in_channels - ic_tile));
        int tile_elements = current_tile * kernel_h * kernel_w;

        // Each thread loads a tile of weights into shared memory, optimizing memory access patterns.
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int idx = tid; idx < tile_elements; idx += blockDim.x * blockDim.y) {
            int local_ic = idx / (kernel_h * kernel_w);
            int rem = idx % (kernel_h * kernel_w);
            int kh = rem / kernel_w;
            int kw = rem % kernel_w;
            int global_ic = ic_tile + local_ic;
            // Weight is stored with shape [in_channels, out_channels, kernel_h, kernel_w]
            // Index: (((global_ic * out_channels) + out_c) * kernel_h + kh) * kernel_w + kw
            shared_weight[idx] = weight[(((global_ic * out_channels) + out_c) * kernel_h + kh) * kernel_w + kw];
        }
        __syncthreads();

        // Iterate over the current tile of input channels
        for (int ic = 0; ic < current_tile; ic++) {
            int global_ic = ic_tile + ic;
            // For each kernel element
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int pos_y = out_y + pad_h - kh;
                    int pos_x = out_x + pad_w - kw;
                    // Check alignment with stride
                    if ((pos_y % stride_h == 0) && (pos_x % stride_w == 0)) {
                        int in_y = pos_y / stride_h;
                        int in_x = pos_x / stride_w;
                        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                            // Input tensor is [batch, in_channels, in_height, in_width]
                            int input_index = ((b * in_channels + global_ic) * in_height + in_y) * in_width + in_x;
                            float input_val = input[input_index];
                            // Get corresponding weight from shared memory
                            int weight_idx = (ic * kernel_h + kh) * kernel_w + kw;
                            float weight_val = shared_weight[weight_idx];
                            sum += input_val * weight_val;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[out_c];
    }
    // Write the output pixel; output shape is [batch, out_channels, out_height, out_width]
    int out_index = ((b * out_channels + out_c) * out_height + out_y) * out_width + out_x;
    output[out_index] = sum;
}


// Host function integrating the CUDA kernel
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    pybind11::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    // Determine bias pointer if bias is provided
    const float* bias_ptr = nullptr;
    torch::Tensor bias;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
        bias_ptr = bias.data_ptr<float>();
    }

    // x: [batch, in_channels, in_height, in_width]
    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    // weight: [in_channels, out_channels, kernel_h, kernel_w]
    int out_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    // Compute output spatial dimensions
    int out_height = (in_height - 1) * stride[0] - 2 * padding[0] + kernel_h;
    int out_width  = (in_width - 1) * stride[1] - 2 * padding[1] + kernel_w;

    // Allocate output tensor
    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    // Define CUDA block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (out_width + blockDim.x - 1) / blockDim.x,
        (out_height + blockDim.y - 1) / blockDim.y,
        batch * out_channels
    );

    // Shared memory size: TILE_IC * kernel_h * kernel_w * sizeof(float)
    size_t sharedMemSize = TILE_IC * kernel_h * kernel_w * sizeof(float);

    // Launch the kernel
    conv_transpose2d_shared_kernel<<<gridDim, blockDim, sharedMemSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with shared memory",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
