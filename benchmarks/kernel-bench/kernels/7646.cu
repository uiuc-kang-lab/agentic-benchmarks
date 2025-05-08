#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handles.h>
#include <ATen/cudnn/Descriptors.h>
#include <cudnn.h>

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
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "Bias must be a CUDA tensor");

    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_depth = input.size(2);
    int64_t in_height = input.size(3);
    int64_t in_width = input.size(4);

    int64_t out_channels = weight.size(0);
    int64_t kernel_d = weight.size(2);
    int64_t kernel_h = weight.size(3);
    int64_t kernel_w = weight.size(4);

    int64_t out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int64_t out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    cudnnHandle_t handle = at::native::getCudnnHandle();

    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t weight_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&weight_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);
    if (bias.defined()) cudnnCreateTensorDescriptor(&bias_desc);

    cudnnDataType_t cudnn_dtype = getCudnnDataType(input.scalar_type());

    int input_dims[5] = {(int)batch_size, (int)in_channels, (int)in_depth, (int)in_height, (int)in_width};
    int input_strides[5] = {(int)(in_channels * in_depth * in_height * in_width),
                           (int)(in_depth * in_height * in_width),
                           (int)(in_height * in_width),
                           (int)in_width, 1};
    cudnnSetTensorNdDescriptor(input_desc, cudnn_dtype, 5, input_dims, input_strides);

    int output_dims[5] = {(int)batch_size, (int)out_channels, (int)out_depth, (int)out_height, (int)out_width};
    int output_strides[5] = {(int)(out_channels * out_depth * out_height * out_width),
                            (int)(out_depth * out_height * out_width),
                            (int)(out_height * out_width),
                            (int)out_width, 1};
    cudnnSetTensorNdDescriptor(output_desc, cudnn_dtype, 5, output_dims, output_strides);

    int filter_dims[5] = {(int)out_channels, (int)(in_channels / groups), (int)kernel_d, (int)kernel_h, (int)kernel_w};
    cudnnSetFilterNdDescriptor(weight_desc, cudnn_dtype, CUDNN_TENSOR_NCHW, 5, filter_dims);

    int padA[3] = {(int)padding, (int)padding, (int)padding};
    int filterStrideA[3] = {(int)stride, (int)stride, (int)stride};
    int dilationA[3] = {(int)dilation, (int)dilation, (int)dilation};
    cudnnSetConvolutionNdDescriptor(conv_desc, 3, padA, filterStrideA, dilationA,
                                   CUDNN_CROSS_CORRELATION, cudnn_dtype);
    cudnnSetConvolutionGroupCount(conv_desc, groups);

    // Set math type to TENSOR_OP to enable Tensor Cores on H100
    cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);

    // Use exhaustive search for best algorithm with specific block size considerations
    const int requestedAlgoCount = 10;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[requestedAlgoCount];
    
    cudnnFindConvolutionForwardAlgorithmEx(
        handle,
        input_desc, input.data_ptr(),
        weight_desc, weight.data_ptr(),
        conv_desc,
        output_desc, output.data_ptr(),
        requestedAlgoCount,
        &returnedAlgoCount,
        perfResults,
        workspace.data_ptr(),
        workspace_size
    );

    // Select the fastest algorithm that fits our memory constraints
    cudnnConvolutionFwdAlgo_t selectedAlgo = perfResults[0].algo;

    // Get workspace size for selected algorithm
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, weight_desc,
                                           conv_desc, output_desc, selectedAlgo,
                                           &workspace_size);

    auto workspace = at::empty({(int64_t)workspace_size}, input.options().dtype(at::kByte));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudnnConvolutionForward(handle, &alpha,
                           input_desc, input.data_ptr(),
                           weight_desc, weight.data_ptr(),
                           conv_desc, selectedAlgo,
                           workspace.data_ptr(), workspace_size,
                           &beta,
                           output_desc, output.data_ptr());

    if (bias.defined()) {
        int bias_dims[5] = {1, (int)out_channels, 1, 1, 1};
        int bias_strides[5] = {(int)out_channels, 1, 1, 1, 1};
        cudnnSetTensorNdDescriptor(bias_desc, cudnn_dtype, 5, bias_dims, bias_strides);
        
        cudnnAddTensor(handle, &alpha,
                       bias_desc, bias.data_ptr(),
                       &alpha,
                       output_desc, output.data_ptr());
    }

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(weight_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    if (bias.defined()) cudnnDestroyTensorDescriptor(bias_desc);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward using cuDNN (CUDA)");
}