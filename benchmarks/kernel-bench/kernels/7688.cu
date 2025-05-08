#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handles.h>
#include <ATen/cudnn/Descriptors.h>
#include <cudnn.h>

__global__ void gpu_convolution(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_depth,
    int in_height, int in_width, int out_channels,
    int kernel_d, int kernel_h, int kernel_w,
    int out_depth, int out_height, int out_width,
    int stride, int padding, int dilation, int groups) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;

    if (tid < total_elements) {
        int batch_idx = tid / (out_channels * out_depth * out_height * out_width);
        int offset = tid % (out_channels * out_depth * out_height * out_width);
        int out_c = offset / (out_depth * out_height * out_width);
        offset = offset % (out_depth * out_height * out_width);
        int out_d = offset / (out_height * out_width);
        offset = offset % (out_height * out_width);
        int out_h = offset / out_width;
        int out_w = offset % out_width;

        float value = 0.0f;
        int group_idx = out_c / (out_channels / groups);
        int in_c_start = group_idx * (in_channels / groups);
        int in_c_end = (group_idx + 1) * (in_channels / groups);

        for (int in_c = in_c_start; in_c < in_c_end; ++in_c) {
            for (int k_d = 0; k_d < kernel_d; ++k_d) {
                for (int k_h = 0; k_h < kernel_h; ++k_h) {
                    for (int k_w = 0; k_w < kernel_w; ++k_w) {
                        int in_d = out_d * stride - padding + k_d * dilation;
                        int in_h = out_h * stride - padding + k_h * dilation;
                        int in_w = out_w * stride - padding + k_w * dilation;
                        if (in_d >= 0 && in_d < in_depth && 
                            in_h >= 0 && in_h < in_height && 
                            in_w >= 0 && in_w < in_width) {
                            int in_idx = batch_idx * in_channels * in_depth * in_height * in_width +
                                       in_c * in_depth * in_height * in_width +
                                       in_d * in_height * in_width +
                                       in_h * in_width + in_w;
                            int w_idx = out_c * (in_channels / groups) * kernel_d * kernel_h * kernel_w +
                                      (in_c - in_c_start) * kernel_d * kernel_h * kernel_w +
                                      k_d * kernel_h * kernel_w +
                                      k_h * kernel_w + k_w;
                            value += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        output[tid] = value;
    }
}

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

    // cuDNN handles and descriptors
    cudnnHandle_t handle = at::native::getCudnnHandle();

    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t weight_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t bias_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&weight_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);
    if (bias.defined()) {
        cudnnCreateTensorDescriptor(&bias_desc);
    }

    // Set tensor descriptors
    cudnnDataType_t cudnn_dtype = getCudnnDataType(input.scalar_type());

    int input_tensor_dim = 5;
    int input_dims[5] = {(int)batch_size, (int)in_channels, (int)in_depth, (int)in_height, (int)in_width};
    int input_strides[5] = {
        (int)(in_channels * in_depth * in_height * in_width),
        (int)(in_depth * in_height * in_width),
        (int)(in_height * in_width),
        (int)(in_width),
        1
    };
    cudnnSetTensorNdDescriptor(input_desc, cudnn_dtype, input_tensor_dim, input_dims, input_strides);

    int output_tensor_dim = 5;
    int output_dims[5] = {(int)batch_size, (int)out_channels, (int)out_depth, (int)out_height, (int)out_width};
    int output_strides[5] = {
        (int)(out_channels * out_depth * out_height * out_width),
        (int)(out_depth * out_height * out_width),
        (int)(out_height * out_width),
        (int)(out_width),
        1
    };
    cudnnSetTensorNdDescriptor(output_desc, cudnn_dtype, output_tensor_dim, output_dims, output_strides);

    int filter_dim = 5;
    int filter_dims[5] = {(int)out_channels, (int)(in_channels / groups), (int)kernel_d, (int)kernel_h, (int)kernel_w};
    cudnnSetFilterNdDescriptor(weight_desc, cudnn_dtype, CUDNN_TENSOR_NCHW, filter_dim, filter_dims);

    int conv_dim = 3;
    int conv_padA[3] = {(int)padding, (int)padding, (int)padding};
    int conv_dilationA[3] = {(int)dilation, (int)dilation, (int)dilation};
    int conv_strideA[3] = {(int)stride, (int)stride, (int)stride};
    cudnnSetConvolutionNdDescriptor(conv_desc, conv_dim, conv_padA, conv_strideA, conv_dilationA,
                                    CUDNN_CROSS_CORRELATION, cudnn_dtype);
    cudnnSetConvolutionGroupCount(conv_desc, groups);

    if (bias.defined()) {
        int bias_dims[5] = {1, (int)out_channels, 1, 1, 1};
        int bias_strides[5] = {(int)out_channels, 1, 1, 1, 1};
        cudnnSetTensorNdDescriptor(bias_desc, cudnn_dtype, 5, bias_dims, bias_strides);
    }

    // Choose the best algorithm for the convolution
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

    // Get workspace size
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle,
                                            input_desc,
                                            weight_desc,
                                            conv_desc,
                                            output_desc,
                                            algo,
                                            &workspace_size);

    // Allocate workspace
    at::Tensor workspace = at::empty({(int64_t)workspace_size}, input.options().dtype(at::kByte));

    // Set up stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudnnSetStream(handle, stream);

    // Perform the convolution in parallel across the batch dimension
    int64_t num_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    gpu_convolution<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth,
        in_height, in_width, out_channels,
        kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation, groups);

    // Add bias if it exists
    if (bias.defined()) {
        cudnnAddTensor(handle,
                       &alpha,
                       bias_desc,
                       bias.data_ptr(),
                       &alpha,
                       output_desc,
                       output.data_ptr());
    }

    // Clean up
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(weight_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    if (bias.defined()) {
        cudnnDestroyTensorDescriptor(bias_desc);
    }

    return output;
}

__global__ void gpu_convolution(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_depth,
    int in_height, int in_width, int out_channels,
    int kernel_d, int kernel_h, int kernel_w,
    int out_depth, int out_height, int out_width,
    int stride, int padding, int dilation, int groups) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;

    if (tid < total_elements) {
        int batch_idx = tid / (out_channels * out_depth * out_height * out_width);
        int offset = tid % (out_channels * out_depth * out_height * out_width);
        int out_c = offset / (out_depth * out_height * out_width);
        offset = offset % (out_depth * out_height * out_width);
        int out_d = offset / (out_height * out_width);
        offset = offset % (out_height * out_width);
        int out_h = offset / out_width;
        int out_w = offset % out_width;

        float value = 0.0;
        for (int in_c = 0; in_c < in_channels / groups; ++in_c) {
            for (int k_d = 0; k_d < kernel_d; ++k_d) {
                for (int k_h = 0; k_h < kernel_h; ++k_h) {
                    for (int k_w = 0; k_w < kernel_w; ++k_w) {
                        int in_d = out_d * stride - padding + k_d * dilation;
                        int in_h = out_h * stride - padding + k_h * dilation;
                        int in_w = out_w * stride - padding + k_w * dilation;
                        if (in_d >= 0 && in_d < in_depth && in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                            value += input[batch_idx * in_channels * in_depth * in_height * in_width + in_c * in_depth * in_height * in_width + in_d * in_height * in_width + in_h * in_width + in_w]
                                   * weight[out_c * (in_channels / groups) * kernel_d * kernel_h * kernel_w + in_c * kernel_d * kernel_h * kernel_w + k_d * kernel_h * kernel_w + k_h * kernel_w + k_w];
                        }
                    }
                }
            }
        }
        output[tid] = value;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D convolution forward using CUDA");
}
