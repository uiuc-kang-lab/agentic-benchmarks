/*
Hybrid 3D Convolution Kernel and Forward Function
This implementation combines a custom CUDA kernel with an optional cuDNN path.
The custom kernel uses a 3D grid mapping to better coalesce memory access compared to a simple 1D iteration,
while the cuDNN branch leverages optimal algorithm selection for larger problem sizes.

Compile this as a PyTorch CUDA extension.
*/

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handles.h>
#include <cudnn.h>

// Define block dimensions for the custom kernel
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 4

// Custom CUDA kernel using 3D grid mapping
// Each thread computes one output element for the 5D tensor with layout [B, C_out, D, H, W]
__global__ void hybrid_conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // pass nullptr if not used
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int kernel_d, int kernel_h, int kernel_w,
    int out_depth, int out_height, int out_width,
    int stride, int padding, int dilation, int groups,
    int grid_d  // number of blocks required to cover out_depth
) {
    // Compute output spatial coordinates from blockIdx and threadIdx
    int w_out = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    int h_out = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;

    // Decode blockIdx.z to obtain batch index, output channel index and a block index in depth dimension.
    int combined = blockIdx.z; // combined index for (B, C_out, depth block)
    int grid_d_total = grid_d; // number of blocks covering depth
    int b = combined / (out_channels * grid_d_total);
    int rem = combined % (out_channels * grid_d_total);
    int c_out = rem / grid_d_total;
    int d_block = rem % grid_d_total;
    int d_out = d_block * BLOCK_SIZE_Z + threadIdx.z;

    // Check boundary conditions
    if (w_out >= out_width || h_out >= out_height || d_out >= out_depth) return;

    float sum = 0.0f;

    // Determine group and channel index mapping
    int group = c_out / (out_channels / groups);
    int in_channels_per_group = in_channels / groups;

    // Loop over input channels for this group
    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        int in_channel = group * in_channels_per_group + ic;
        // Loop over kernel volume
        for (int kd = 0; kd < kernel_d; kd++) {
            int d_in = d_out * stride - padding + kd * dilation;
            if (d_in < 0 || d_in >= in_depth) continue;
            for (int kh = 0; kh < kernel_h; kh++) {
                int h_in = h_out * stride - padding + kh * dilation;
                if (h_in < 0 || h_in >= in_height) continue;
                for (int kw = 0; kw < kernel_w; kw++) {
                    int w_in = w_out * stride - padding + kw * dilation;
                    if (w_in < 0 || w_in >= in_width) continue;

                    int input_idx = ((b * in_channels + in_channel) * in_depth + d_in) * in_height * in_width
                                    + h_in * in_width + w_in;
                    int weight_idx = ((c_out * in_channels_per_group + ic) * kernel_d + kd) * kernel_h * kernel_w
                                     + kh * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    int output_idx = ((b * out_channels + c_out) * out_depth + d_out) * out_height * out_width
                     + h_out * out_width + w_out;
    output[output_idx] = sum;
}

// Helper function to map at::ScalarType to cuDNN dataType
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

// Hybrid forward: selects between custom kernel and cuDNN path
// Setting use_cudnn to true leverages cuDNN's optimal algorithms
at::Tensor hybrid_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool use_cudnn  // if true, use cuDNN, else use custom kernel
) {
    auto bias = bias_opt.value_or(at::Tensor());
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

    // Input dimensions (B x in_channels x D x H x W)
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_depth = input.size(2);
    int64_t in_height = input.size(3);
    int64_t in_width = input.size(4);

    // Weight dimensions (out_channels x (in_channels/groups) x kD x kH x kW)
    int64_t out_channels = weight.size(0);
    int64_t kernel_d = weight.size(2);
    int64_t kernel_h = weight.size(3);
    int64_t kernel_w = weight.size(4);

    // Calculate output dimensions
    int64_t out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int64_t out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto options = input.options();
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, options);

    if (use_cudnn) {
        // cuDNN based convolution (similar to Kernel 2)
        cudnnHandle_t handle = at::native::getCudnnHandle();
        cudnnTensorDescriptor_t input_desc, output_desc, bias_desc = nullptr;
        cudnnFilterDescriptor_t weight_desc;
        cudnnConvolutionDescriptor_t conv_desc;

        cudnnCreateTensorDescriptor(&input_desc);
        cudnnCreateTensorDescriptor(&output_desc);
        cudnnCreateFilterDescriptor(&weight_desc);
        cudnnCreateConvolutionDescriptor(&conv_desc);
        if (bias.defined()) {
            cudnnCreateTensorDescriptor(&bias_desc);
        }

        cudnnDataType_t cudnn_dtype = getCudnnDataType(input.scalar_type());
        int dims = 5;
        int input_dims[5] = { (int)batch_size, (int)in_channels, (int)in_depth, (int)in_height, (int)in_width };
        int input_strides[5] = {
            (int)(in_channels * in_depth * in_height * in_width),
            (int)(in_depth * in_height * in_width),
            (int)(in_height * in_width),
            (int)(in_width),
            1
        };
        cudnnSetTensorNdDescriptor(input_desc, cudnn_dtype, dims, input_dims, input_strides);

        int output_dims[5] = { (int)batch_size, (int)out_channels, (int)out_depth, (int)out_height, (int)out_width };
        int output_strides[5] = {
            (int)(out_channels * out_depth * out_height * out_width),
            (int)(out_depth * out_height * out_width),
            (int)(out_height * out_width),
            (int)(out_width),
            1
        };
        cudnnSetTensorNdDescriptor(output_desc, cudnn_dtype, dims, output_dims, output_strides);

        int filter_dims[5] = { (int)out_channels, (int)(in_channels / groups), (int)kernel_d, (int)kernel_h, (int)kernel_w };
        cudnnSetFilterNdDescriptor(weight_desc, cudnn_dtype, CUDNN_TENSOR_NCHW, 5, filter_dims);

        int conv_dim = 3;
        int padA[3] = { (int)padding, (int)padding, (int)padding };
        int strideA[3] = { (int)stride, (int)stride, (int)stride };
        int dilationA[3] = { (int)dilation, (int)dilation, (int)dilation };
        cudnnSetConvolutionNdDescriptor(conv_desc, conv_dim, padA, strideA, dilationA,
                                        CUDNN_CROSS_CORRELATION, cudnn_dtype);
        cudnnSetConvolutionGroupCount(conv_desc, groups);

        if (bias.defined()) {
            int bias_dims[5] = { 1, (int)out_channels, 1, 1, 1 };
            int bias_strides[5] = { (int)out_channels, 1, 1, 1, 1 };
            cudnnSetTensorNdDescriptor(bias_desc, cudnn_dtype, 5, bias_dims, bias_strides);
        }

        cudnnConvolutionFwdAlgoPerf_t algoPerf;
        int returnedAlgoCount;
        cudnnFindConvolutionForwardAlgorithm(
            handle,
            input_desc,
            weight_desc,
            conv_desc,
            output_desc,
            1,
            &returnedAlgoCount,
            &algoPerf
        );
        cudnnConvolutionFwdAlgo_t algo = algoPerf.algo;

        size_t workspace_size = 0;
        cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                input_desc,
                                                weight_desc,
                                                conv_desc,
                                                output_desc,
                                                algo,
                                                &workspace_size);

        at::Tensor workspace = at::empty({(int64_t)workspace_size}, input.options().dtype(at::kByte));

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudnnConvolutionForward(handle,
                                &alpha,
                                input_desc,
                                input.data_ptr(),
                                weight_desc,
                                weight.data_ptr(),
                                conv_desc,
                                algo,
                                workspace.data_ptr(),
                                workspace_size,
                                &beta,
                                output_desc,
                                output.data_ptr());

        if (bias.defined()) {
            cudnnAddTensor(handle,
                           &alpha,
                           bias_desc,
                           bias.data_ptr(),
                           &alpha,
                           output_desc,
                           output.data_ptr());
        }

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyFilterDescriptor(weight_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        if (bias.defined()) {
            cudnnDestroyTensorDescriptor(bias_desc);
        }

        return output;
    } else {
        // Use the custom CUDA kernel
        dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
        int grid_x = (out_width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
        int grid_y = (out_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
        int grid_d = (out_depth + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
        // grid.z is used to cover (batch_size x out_channels x depth blocks)
        int grid_z = batch_size * out_channels * grid_d;
        dim3 grid(grid_x, grid_y, grid_z);

        // Launch the kernel
        hybrid_conv3d_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            in_depth, in_height, in_width,
            kernel_d, kernel_h, kernel_w,
            out_depth, out_height, out_width,
            stride, padding, dilation, groups,
            grid_d
        );
        return output;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hybrid_forward", &hybrid_forward, "Hybrid 3D convolution forward (CUDA) with optional cuDNN");
}
