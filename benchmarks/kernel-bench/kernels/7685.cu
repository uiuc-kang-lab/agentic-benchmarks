#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handles.h>
#include <ATen/cudnn/Descriptors.h>
#include <cudnn.h>

// Helper function to map at::ScalarType to cudnnDataType_t
cudnnDataType_t getCudnnDataType(at::ScalarType type) {
    switch (type) {
        case at::ScalarType::Float:
            return CUDNN_DATA_FLOAT;
        case at::ScalarType::Double:
            return CUDNN_DATA_DOUBLE;
        case at::ScalarType::Half:
            return CUDNN_DATA_HALF;
        default:
            TORCH_CHECK(false, "Unsupported data type for cuDNN");
    }
}

__global__ void conv3dSharedKernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int in_depth, int in_height, int in_width,
    int out_channels, int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int dilation,
    int out_depth, int out_height, int out_width
) {
    extern __shared__ float shared_input[];

    // Calculate thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // Load data into shared memory
    int shared_size = (kernel_d + 2 * padding) * (kernel_h + 2 * padding) * (kernel_w + 2 * padding);
    int shared_index = tz * (kernel_h + 2 * padding) * (kernel_w + 2 * padding) + ty * (kernel_w + 2 * padding) + tx;
    if (shared_index < shared_size) {
        shared_input[shared_index] = input[shared_index];
    }
    __syncthreads();

    // Compute convolution
    if (bx < out_width && by < out_height && bz < out_depth) {
        float value = 0.0f;
        for (int c = 0; c < out_channels; ++c) {
            for (int z = 0; z < kernel_d; ++z) {
                for (int y = 0; y < kernel_h; ++y) {
                    for (int x = 0; x < kernel_w; ++x) {
                        int input_idx = ((bz * stride + z * dilation) * in_height + (by * stride + y * dilation)) * in_width + (bx * stride + x * dilation);
                        int weight_idx = ((c * kernel_d + z) * kernel_h + y) * kernel_w + x;
                        value += shared_input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        int output_idx = ((bz * out_height + by) * out_width + bx);
        output[output_idx] = value;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto bias = bias_opt.value_or(at::Tensor());
    // Ensure inputs are on CUDA
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "Bias must be a CUDA tensor");

    // Get input dimensions
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_depth = input.size(2);
    int64_t in_height = input.size(3);
    int64_t in_width = input.size(4);

    // Get weight dimensions
    int64_t out_channels = weight.size(0);
    int64_t kernel_d = weight.size(2);
    int64_t kernel_h = weight.size(3);
    int64_t kernel_w = weight.size(4);

    // Calculate output dimensions
    int64_t out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int64_t out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    // Prepare output tensor
    auto options = input.options();
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, options);

    dim3 threads(kernel_w, kernel_h, kernel_d);
    dim3 blocks(out_width / threads.x, out_height / threads.y, out_depth / threads.z);

    int shared_mem_size = (kernel_d + 2 * padding) * (kernel_h + 2 * padding) * (kernel_w + 2 * padding);
    conv3dSharedKernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        in_depth, in_height, in_width,
        out_channels, kernel_d, kernel_h, kernel_w,
        stride, padding, dilation,
        out_depth, out_height, out_width
    );

    if (bias.defined()) {
        const float alpha = 1.0f;
        cudnnHandle_t handle = at::native::getCudnnHandle();

        cudnnTensorDescriptor_t bias_desc;
        cudnnCreateTensorDescriptor(&bias_desc);
        int bias_dims[5] = {1, (int)out_channels, 1, 1, 1};
        int bias_strides[5] = {(int)out_channels, 1, 1, 1, 1};
        cudnnSetTensorNdDescriptor(bias_desc, getCudnnDataType(input.scalar_type()), 5, bias_dims, bias_strides);

        cudnnTensorDescriptor_t output_desc;
        cudnnCreateTensorDescriptor(&output_desc);
        int output_dims[5] = {(int)batch_size, (int)out_channels, (int)out_depth, (int)out_height, (int)out_width};
        int output_strides[5] = {
            (int)(out_channels * out_depth * out_height * out_width),
            (int)(out_depth * out_height * out_width),
            (int)(out_height * out_width),
            (int)(out_width),
            1
        };
        cudnnSetTensorNdDescriptor(output_desc, getCudnnDataType(input.scalar_type()), 5, output_dims, output_strides);

        cudnnAddTensor(handle,
                       &alpha,
                       bias_desc,
                       bias.data_ptr(),
                       &alpha,
                       output_desc,
                       output.data_ptr());

        cudnnDestroyTensorDescriptor(bias_desc);
        cudnnDestroyTensorDescriptor(output_desc);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward using shared memory (CUDA)");
}