#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel definition
__global__ void transposed_conv3d_kernel(const float* __restrict__ input, const float* __restrict__ filter, float* output, 
                 int input_dim0, int input_dim1, int input_dim2,
                 int filter_dim0, int filter_dim1, int filter_dim2,
                 int output_dim0, int output_dim1, int output_dim2,
                 int stride0, int stride1, int stride2,
                 int padding0, int padding1, int padding2) {
    // Calculate the 3D coordinates of the output
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < output_dim0 && y < output_dim1 && z < output_dim2) {
        float value = 0.0f;
        for (int f0 = 0; f0 < filter_dim0; ++f0) {
            for (int f1 = 0; f1 < filter_dim1; ++f1) {
                for (int f2 = 0; f2 < filter_dim2; ++f2) {
                    int ix = x * stride0 - padding0 + f0;
                    int iy = y * stride1 - padding1 + f1;
                    int iz = z * stride2 - padding2 + f2;
                    if (ix >= 0 && ix < input_dim0 && iy >= 0 && iy < input_dim1 && iz >= 0 && iz < input_dim2) {
                        // Use __ldg for read-only data access
                        float input_val = __ldg(&input[(ix * input_dim1 * input_dim2) + (iy * input_dim2) + iz]);
                        float weight_val = __ldg(&filter[(f0 * filter_dim1 * filter_dim2) + (f1 * filter_dim2) + f2]);
                        value += input_val * weight_val;
                    }
                }
            }
        }
        output[(x * output_dim1 * output_dim2) + (y * output_dim2) + z] = value;
    }
}

// Forward function to launch kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto input_dim0 = x.size(2);
    const auto input_dim1 = x.size(3);
    const auto input_dim2 = x.size(4);

    const auto out_channels = weight.size(0);
    const auto filter_dim0 = weight.size(2);
    const auto filter_dim1 = weight.size(3);
    const auto filter_dim2 = weight.size(4);

    const int output_dim0 = (input_dim0 - 1) * stride[0] - 2 * padding[0] + filter_dim0 + output_padding[0];
    const int output_dim1 = (input_dim1 - 1) * stride[1] - 2 * padding[1] + filter_dim1 + output_padding[1];
    const int output_dim2 = (input_dim2 - 1) * stride[2] - 2 * padding[2] + filter_dim2 + output_padding[2];

    // Prepare the output tensor
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, out_channels, output_dim0, output_dim1, output_dim2}, options);

    // Define CUDA kernel launch dimensions
    dim3 threads_per_block(8, 8, 8);
    dim3 blocks_per_grid((output_dim0 + threads_per_block.x - 1) / threads_per_block.x,
                         (output_dim1 + threads_per_block.y - 1) / threads_per_block.y,
                         (output_dim2 + threads_per_block.z - 1) / threads_per_block.z);

    for (int n = 0; n < batch_size; ++n) {
        // Launch CUDA kernel
        transposed_conv3d_kernel<<<blocks_per_grid, threads_per_block>>>(
            x[n].data_ptr<float>(),
            weight.data_ptr<float>(),
            output[n].data_ptr<float>(),
            input_dim0, input_dim1, input_dim2,
            filter_dim0, filter_dim1, filter_dim2,
            output_dim0, output_dim1, output_dim2,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2]
        );
    }
    return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward function optimized",
          py::arg("x"),
          py::arg("weight"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}