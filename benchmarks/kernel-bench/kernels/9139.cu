#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_height,
    const int kernel_width,
    const int out_height,
    const int out_width,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w
) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    
    // Calculate output position
    const int out_y = bid / out_width;
    const int out_x = bid % out_width;
    
    // Initialize shared memory
    shared_mem[tid] = 0.0f;
    __syncthreads();
    
    // Compute input window boundaries
    const int in_y_start = (out_y + pad_h - kernel_height + 1 + stride_h - 1) / stride_h;
    const int in_x_start = (out_x + pad_w - kernel_width + 1 + stride_w - 1) / stride_w;
    const int in_y_end = (out_y + pad_h + stride_h - 1) / stride_h;
    const int in_x_end = (out_x + pad_w + stride_w - 1) / stride_w;
    
    // Parallel reduction across input channels
    for (int b = 0; b < batch_size; b++) {
        for (int oc = wid; oc < out_channels; oc += blockDim.x / 32) {
            float sum = 0.0f;
            
            for (int ic = lane; ic < in_channels; ic += 32) {
                for (int ky = 0; ky < kernel_height; ky++) {
                    for (int kx = 0; kx < kernel_width; kx++) {
                        const int in_y = in_y_start + ky;
                        const int in_x = in_x_start + kx;
                        
                        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                            const int in_idx = ((b * in_channels + ic) * in_height + in_y) * in_width + in_x;
                            const int w_idx = ((oc * in_channels + ic) * kernel_height + ky) * kernel_width + kx;
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
            
            // Warp reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            if (lane == 0) {
                const int out_idx = ((b * out_channels + oc) * out_height + out_y) * out_width + out_x;
                output[out_idx] = sum;
            }
        }
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    const auto out_channels = weight.size(1);
    const auto kernel_height = weight.size(2);
    const auto kernel_width = weight.size(3);
    
    const auto out_height = (in_height - 1) * stride[0] - 2 * padding[0] + kernel_height;
    const auto out_width = (in_width - 1) * stride[1] - 2 * padding[1] + kernel_width;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    const int threads_per_block = 256;
    const int blocks = out_height * out_width;
    const int shared_mem_size = threads_per_block * sizeof(float);
    
    conv_transpose2d_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}