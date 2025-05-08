#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handles.h>
#include <ATen/cudnn/Descriptors.h>
#include <cudnn.h>

// Define block size for our custom kernel (tuned for modern GPUs, e.g., H100)
#define BLOCK_SIZE 1024

//---------------------------------------------------------------------------
// Custom CUDA kernel for 3D convolution (inspired by Kernel 1)
// This kernel uses loop unrolling and assigns each thread to compute
// multiple output elements for a given output channel and batch
//---------------------------------------------------------------------------
__global__ void hybrid_conv3d_custom_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation
) {
    // Each block corresponds to one output channel and one batch sample
    const int oc = blockIdx.x;
    const int batch_id = blockIdx.y;

    // Flattened index for the output element in the (out_depth*out_height*out_width) volume
    int tid = threadIdx.x;
    int total_elements = out_depth * out_height * out_width;

    for (int idx = tid; idx < total_elements; idx += blockDim.x) {
        // Decode the 3D coordinates from the linear index
        int od = idx / (out_height * out_width);
        int tmp = idx % (out_height * out_width);
        int oh = tmp / out_width;
        int ow = tmp % out_width;

        float sum = 0.0f;
        // Loop over input channels and kernel dimensions (using unroll pragmas for performance)
        #pragma unroll
        for (int ic = 0; ic < in_channels; ++ic) {
            #pragma unroll
            for (int kd = 0; kd < kernel_d; ++kd) {
                int id = od * stride - padding + kd * dilation;
                if (id >= 0 && id < in_depth) {
                    #pragma unroll
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        int ih = oh * stride - padding + kh * dilation;
                        if (ih >= 0 && ih < in_height) {
                            #pragma unroll
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int iw = ow * stride - padding + kw * dilation;
                                if (iw >= 0 && iw < in_width) {
                                    int input_idx = (((batch_id * in_channels + ic) * in_depth + id) * in_height + ih) * in_width + iw;
                                    int weight_idx = (((oc * in_channels + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        if (bias) {
            sum += bias[oc];
        }
        int output_idx = (((batch_id * out_channels + oc) * out_depth + od) * out_height + oh) * out_width + ow;
        output[output_idx] = sum;
    }
}

//---------------------------------------------------------------------------
// Custom forward function that launches the custom kernel
//---------------------------------------------------------------------------
at::Tensor hybrid_conv3d_forward_custom(const at::Tensor& input,
                                         const at::Tensor& weight,
                                         const at::Tensor& bias,
                                         int stride,
                                         int padding,
                                         int dilation) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    // Each block handles one output channel and one batch sample
    dim3 grid(out_channels, batch_size);
    int num_threads = BLOCK_SIZE;

    hybrid_conv3d_custom_kernel<<<grid, num_threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation
    );

    return output;
}

//---------------------------------------------------------------------------
// Hybrid forward function that combines cuDNN (Kernel 2) and the custom kernel
//---------------------------------------------------------------------------
// This function dispatches to cuDNN when either groups != 1 or the user opts in (use_cudnn == true).
// Otherwise, it uses the lightweight custom kernel. This allows leveraging cuDNN's auto-tuning
// for complex cases while avoiding its overhead for simpler workloads.
//---------------------------------------------------------------------------

at::Tensor hybrid_conv3d_forward(const at::Tensor& input,
                                 const at::Tensor& weight,
                                 const c10::optional<at::Tensor>& bias_opt,
                                 int stride,
                                 int padding,
                                 int dilation,
                                 int groups,
                                 bool use_cudnn = true) {
    auto bias = bias_opt.value_or(at::Tensor());
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    
    // Use cuDNN path if groups != 1 or the user chooses to use cuDNN
    if (groups != 1 || use_cudnn) {
        //-----------------------------------------
        // cuDNN-based convolution implementation
        //-----------------------------------------
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

        auto options = input.options();
        auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, options);

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

        // Assume float data type for simplicity
        cudnnDataType_t cudnn_dtype = CUDNN_DATA_FLOAT;
        int input_dims[5] = { (int)batch_size, (int)in_channels, (int)in_depth, (int)in_height, (int)in_width };
        int input_strides[5] = { 
            (int)(in_channels * in_depth * in_height * in_width),
            (int)(in_depth * in_height * in_width),
            (int)(in_height * in_width),
            (int)(in_width),
            1
        };
        cudnnSetTensorNdDescriptor(input_desc, cudnn_dtype, 5, input_dims, input_strides);

        int output_dims[5] = { (int)batch_size, (int)out_channels, (int)out_depth, (int)out_height, (int)out_width };
        int output_strides[5] = {
            (int)(out_channels * out_depth * out_height * out_width),
            (int)(out_depth * out_height * out_width),
            (int)(out_height * out_width),
            (int)(out_width),
            1
        };
        cudnnSetTensorNdDescriptor(output_desc, cudnn_dtype, 5, output_dims, output_strides);

        int filter_dims[5] = { (int)out_channels, (int)in_channels, (int)kernel_d, (int)kernel_h, (int)kernel_w };
        cudnnSetFilterNdDescriptor(weight_desc, cudnn_dtype, CUDNN_TENSOR_NCHW, 5, filter_dims);

        int conv_dim = 3;
        int padA[3] = { padding, padding, padding };
        int strideA[3] = { stride, stride, stride };
        int dilationA[3] = { dilation, dilation, dilation };
        cudnnSetConvolutionNdDescriptor(conv_desc, conv_dim, padA, strideA, dilationA,
                                        CUDNN_CROSS_CORRELATION, cudnn_dtype);
        cudnnSetConvolutionGroupCount(conv_desc, groups);

        if (bias.defined()) {
            int bias_dims[5] = { 1, (int)out_channels, 1, 1, 1 };
            int bias_strides[5] = { (int)out_channels, 1, 1, 1, 1 };
            cudnnSetTensorNdDescriptor(bias_desc, cudnn_dtype, 5, bias_dims, bias_strides);
        }

        // Choose the best convolution algorithm
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
                                input.data_ptr<float>(),
                                weight_desc,
                                weight.data_ptr<float>(),
                                conv_desc,
                                algo,
                                workspace.data_ptr(),
                                workspace_size,
                                &beta,
                                output_desc,
                                output.data_ptr<float>());

        if (bias.defined()) {
            cudnnAddTensor(handle,
                           &alpha,
                           bias_desc,
                           bias.data_ptr<float>(),
                           &alpha,
                           output_desc,
                           output.data_ptr<float>());
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
        // Use custom kernel when cuDNN is not selected
        return hybrid_conv3d_forward_custom(input, weight, bias, stride, padding, dilation);
    }
}

//---------------------------------------------------------------------------
// Pybind11 binding
//---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hybrid_conv3d_forward, "Hybrid 3D convolution forward (cuDNN and custom kernel)");
}
