#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

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
            return CUDNN_DATA_FLOAT; // Unreachable
    }
}

// Forward function that splits the input batch into chunks and processes them asynchronously using CUDA streams
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
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

    // Get input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Get weight dimensions
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Compute output dimensions
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    // Prepare output tensor
    auto options = input.options();
    at::Tensor output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, options);

    // Number of streams to overlap computation and memory operations
    int num_streams = 2;
    int sub_batch = (batch_size + num_streams - 1) / num_streams;

    // Create vectors for CUDA streams and cuDNN handles
    std::vector<cudaStream_t> streams(num_streams);
    std::vector<cudnnHandle_t> handles(num_streams);

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudnnCreate(&handles[i]);
        cudnnSetStream(handles[i], streams[i]);
    }

    // Process each sub-batch asynchronously
    for (int i = 0; i < num_streams; i++) {
        int b_start = i * sub_batch;
        int current_batch = std::min(sub_batch, batch_size - b_start);
        if (current_batch <= 0) break;

        // Create tensor descriptors for the sub-batch input
        int in_dims[5] = { current_batch, in_channels, in_depth, in_height, in_width };
        int in_strides[5] = { in_channels * in_depth * in_height * in_width,
                              in_depth * in_height * in_width,
                              in_height * in_width,
                              in_width,
                              1 };
        cudnnTensorDescriptor_t input_desc;
        cudnnCreateTensorDescriptor(&input_desc);
        cudnnSetTensorNdDescriptor(input_desc, getCudnnDataType(input.scalar_type()), 5, in_dims, in_strides);

        // Create tensor descriptor for the sub-batch output
        int out_dims[5] = { current_batch, out_channels, out_depth, out_height, out_width };
        int out_strides[5] = { out_channels * out_depth * out_height * out_width,
                               out_depth * out_height * out_width,
                               out_height * out_width,
                               out_width,
                               1 };
        cudnnTensorDescriptor_t output_desc;
        cudnnCreateTensorDescriptor(&output_desc);
        cudnnSetTensorNdDescriptor(output_desc, getCudnnDataType(input.scalar_type()), 5, out_dims, out_strides);

        // Create filter descriptor (same for all sub-batches)
        cudnnFilterDescriptor_t weight_desc;
        cudnnCreateFilterDescriptor(&weight_desc);
        int filter_dims[5] = { out_channels, in_channels / groups, kernel_d, kernel_h, kernel_w };
        cudnnSetFilterNdDescriptor(weight_desc, getCudnnDataType(input.scalar_type()), CUDNN_TENSOR_NCHW, 5, filter_dims);

        // Create convolution descriptor
        cudnnConvolutionDescriptor_t conv_desc;
        cudnnCreateConvolutionDescriptor(&conv_desc);
        int conv_padA[3] = { (int)padding, (int)padding, (int)padding };
        int conv_strideA[3] = { (int)stride, (int)stride, (int)stride };
        int conv_dilationA[3] = { (int)dilation, (int)dilation, (int)dilation };
        cudnnSetConvolutionNdDescriptor(conv_desc, 3, conv_padA, conv_strideA, conv_dilationA,
                                        CUDNN_CROSS_CORRELATION, getCudnnDataType(input.scalar_type()));
        cudnnSetConvolutionGroupCount(conv_desc, groups);

        // If bias is provided, create bias descriptor
        cudnnTensorDescriptor_t bias_desc;
        if (bias.defined()) {
            cudnnCreateTensorDescriptor(&bias_desc);
            int bias_dims[5] = {1, out_channels, 1, 1, 1};
            int bias_strides[5] = { out_channels, 1, 1, 1, 1 };
            cudnnSetTensorNdDescriptor(bias_desc, getCudnnDataType(input.scalar_type()), 5, bias_dims, bias_strides);
        }

        // Choose the best convolution algorithm for this sub-batch
        cudnnConvolutionFwdAlgoPerf_t algoPerf;
        int returnedAlgoCount;
        cudnnFindConvolutionForwardAlgorithm(handles[i], input_desc, weight_desc,
                                             conv_desc, output_desc, 1, &returnedAlgoCount, &algoPerf);
        cudnnConvolutionFwdAlgo_t algo = algoPerf.algo;

        // Get workspace size
        size_t workspace_size = 0;
        cudnnGetConvolutionForwardWorkspaceSize(handles[i], input_desc, weight_desc, conv_desc, output_desc, algo, &workspace_size);
        at::Tensor workspace = at::empty({ (int64_t)workspace_size }, input.options().dtype(at::kByte));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Calculate pointers for the current sub-batch
        const float* input_ptr = input.data_ptr<float>() + b_start * in_channels * in_depth * in_height * in_width;
        float* output_ptr = output.data_ptr<float>() + b_start * out_channels * out_depth * out_height * out_width;

        // Launch the convolution asynchronously on the assigned stream
        cudnnConvolutionForward(handles[i],
                                &alpha,
                                input_desc,
                                input_ptr,
                                weight_desc,
                                weight.data_ptr<float>(),
                                conv_desc,
                                algo,
                                workspace.data_ptr(),
                                workspace_size,
                                &beta,
                                output_desc,
                                output_ptr);

        // If bias is defined, add it
        if (bias.defined()) {
            cudnnAddTensor(handles[i],
                           &alpha,
                           bias_desc,
                           bias.data_ptr<float>(),
                           &alpha,
                           output_desc,
                           output_ptr);
            cudnnDestroyTensorDescriptor(bias_desc);
        }

        // Clean up descriptors for this sub-batch
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyFilterDescriptor(weight_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
    }

    // Synchronize all streams and destroy handles/streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudnnDestroy(handles[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with asynchronous pipeline using CUDA streams and cuDNN");
}
