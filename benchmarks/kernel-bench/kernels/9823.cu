#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel: same as reference but operates on a sub-batch
__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    int total_elements = batch_size * out_channels * output_h * output_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int w_out = idx % output_w;
    int temp = idx / output_w;
    int h_out = temp % output_h;
    temp /= output_h;
    int oc = temp % out_channels;
    int b = temp / out_channels;  // b is relative to the sub-batch

    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    float sum = 0.0f;
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = h_out * stride + kh - padding;
            int w_in = w_out * stride + kw - padding;
            
            if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                int input_idx = b * (in_channels * input_h * input_w)
                              + in_ch * (input_h * input_w)
                              + h_in * input_w
                              + w_in;
                
                int weight_idx = in_ch * (channels_per_group * kernel_size * kernel_size)
                               + weight_ch * (kernel_size * kernel_size)
                               + kh * kernel_size
                               + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    output[b * (out_channels * output_h * output_w) +
           oc * (output_h * output_w) +
           h_out * output_w +
           w_out] = sum;
}


// Forward function that partitions the batch across multiple CUDA streams
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Partition the batch into chunks to overlap execution using CUDA streams
    int num_streams = std::min(batch_size, 4); // Use up to 4 streams
    int chunk_size = (batch_size + num_streams - 1) / num_streams;
    
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int threads = 256;
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    for (int i = 0; i < num_streams; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, batch_size);
        int local_batch = end - start;
        if (local_batch <= 0) continue;
        
        int total_elements = local_batch * out_channels * output_h * output_w;
        int blocks = (total_elements + threads - 1) / threads;
        
        // Offset the pointers to process only the current batch slice
        const float* local_input_ptr = input_ptr + start * in_channels * input_h * input_w;
        float* local_output_ptr = output_ptr + start * out_channels * output_h * output_w;
        
        depthwise_conv2d_kernel<<<blocks, threads, 0, streams[i]>>>(
            local_input_ptr,
            weight_ptr,
            bias_ptr,
            local_output_ptr,
            local_batch,
            in_channels,
            input_h,
            input_w,
            out_channels,
            output_h,
            output_w,
            kernel_size,
            stride,
            padding,
            channels_per_group
        );
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution with CUDA Streams",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
