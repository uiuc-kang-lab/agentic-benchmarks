#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handles.h>
#include <ATen/cudnn/Descriptors.h>
#include <cudnn.h>
#include <vector>

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

    const int num_streams = 4; // Number of parallel streams
    std::vector<cudaStream_t> streams(num_streams);
    std::vector<cudnnHandle_t> handles(num_streams);
    
    // Create streams and cuDNN handles
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudnnCreate(&handles[i]);
        cudnnSetStream(handles[i], streams[i]);
    }

    // Get dimensions
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_depth = input.size(2);
    int64_t in_height = input.size(3);
    int64_t in_width = input.size(4);
    
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

    // Calculate chunk size for batch dimension
    int64_t chunk_size = (batch_size + num_streams - 1) / num_streams;

    std::vector<cudnnTensorDescriptor_t> input_descs(num_streams);
    std::vector<cudnnTensorDescriptor_t> output_descs(num_streams);
    std::vector<cudnnFilterDescriptor_t> weight_descs(num_streams);
    std::vector<cudnnConvolutionDescriptor_t> conv_descs(num_streams);
    std::vector<cudnnTensorDescriptor_t> bias_descs(num_streams);

    for (int i = 0; i < num_streams; i++) {
        cudnnCreateTensorDescriptor(&input_descs[i]);
        cudnnCreateTensorDescriptor(&output_descs[i]);
        cudnnCreateFilterDescriptor(&weight_descs[i]);
        cudnnCreateConvolutionDescriptor(&conv_descs[i]);
        if (bias.defined()) {
            cudnnCreateTensorDescriptor(&bias_descs[i]);
        }
    }

    // Process chunks in parallel streams
    for (int i = 0; i < num_streams; i++) {
        int64_t start_idx = i * chunk_size;
        int64_t end_idx = std::min(start_idx + chunk_size, batch_size);
        if (start_idx >= batch_size) break;

        int64_t current_batch_size = end_idx - start_idx;
        
        auto input_chunk = input.narrow(0, start_idx, current_batch_size);
        auto output_chunk = output.narrow(0, start_idx, current_batch_size);

        cudnnDataType_t cudnn_dtype = getCudnnDataType(input.scalar_type());

        // Set descriptors for current chunk
        int input_dims[5] = {(int)current_batch_size, (int)in_channels, (int)in_depth, (int)in_height, (int)in_width};
        int input_strides[5] = {
            (int)(in_channels * in_depth * in_height * in_width),
            (int)(in_depth * in_height * in_width),
            (int)(in_height * in_width),
            (int)in_width,
            1
        };
        cudnnSetTensorNdDescriptor(input_descs[i], cudnn_dtype, 5, input_dims, input_strides);

        int output_dims[5] = {(int)current_batch_size, (int)out_channels, (int)out_depth, (int)out_height, (int)out_width};
        int output_strides[5] = {
            (int)(out_channels * out_depth * out_height * out_width),
            (int)(out_depth * out_height * out_width),
            (int)(out_height * out_width),
            (int)out_width,
            1
        };
        cudnnSetTensorNdDescriptor(output_descs[i], cudnn_dtype, 5, output_dims, output_strides);

        int filter_dims[5] = {(int)out_channels, (int)(in_channels / groups), (int)kernel_d, (int)kernel_h, (int)kernel_w};
        cudnnSetFilterNdDescriptor(weight_descs[i], cudnn_dtype, CUDNN_TENSOR_NCHW, 5, filter_dims);

        int conv_padA[3] = {(int)padding, (int)padding, (int)padding};
        int conv_dilationA[3] = {(int)dilation, (int)dilation, (int)dilation};
        int conv_strideA[3] = {(int)stride, (int)stride, (int)stride};
        cudnnSetConvolutionNdDescriptor(conv_descs[i], 3, conv_padA, conv_strideA, conv_dilationA,
                                      CUDNN_CROSS_CORRELATION, cudnn_dtype);
        cudnnSetConvolutionGroupCount(conv_descs[i], groups);

        // Find best algorithm for current configuration
        cudnnConvolutionFwdAlgoPerf_t algo_perf;
        int returned_algo_count;
        cudnnFindConvolutionForwardAlgorithm(
            handles[i],
            input_descs[i],
            weight_descs[i],
            conv_descs[i],
            output_descs[i],
            1,
            &returned_algo_count,
            &algo_perf
        );

        // Allocate workspace
        size_t workspace_size = 0;
        cudnnGetConvolutionForwardWorkspaceSize(
            handles[i],
            input_descs[i],
            weight_descs[i],
            conv_descs[i],
            output_descs[i],
            algo_perf.algo,
            &workspace_size
        );

        auto workspace = at::empty({(int64_t)workspace_size}, input.options().dtype(at::kByte));

        // Execute convolution
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudnnConvolutionForward(
            handles[i],
            &alpha,
            input_descs[i],
            input_chunk.data_ptr(),
            weight_descs[i],
            weight.data_ptr(),
            conv_descs[i],
            algo_perf.algo,
            workspace.data_ptr(),
            workspace_size,
            &beta,
            output_descs[i],
            output_chunk.data_ptr()
        );

        if (bias.defined()) {
            cudnnAddTensor(
                handles[i],
                &alpha,
                bias_descs[i],
                bias.data_ptr(),
                &alpha,
                output_descs[i],
                output_chunk.data_ptr()
            );
        }
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudnnDestroyTensorDescriptor(input_descs[i]);
        cudnnDestroyTensorDescriptor(output_descs[i]);
        cudnnDestroyFilterDescriptor(weight_descs[i]);
        cudnnDestroyConvolutionDescriptor(conv_descs[i]);
        if (bias.defined()) {
            cudnnDestroyTensorDescriptor(bias_descs[i]);
        }
        cudnnDestroy(handles[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward using cuDNN with streams (CUDA)");
}