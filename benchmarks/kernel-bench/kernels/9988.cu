#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Standard depthwise convolution kernel, unchanged in computation
__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,        // local (sub-batch) size
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    extern __shared__ float shared_mem[];
    
    // Divide shared memory into sections for input tile and weights
    float* shared_input = shared_mem;
    float* shared_weights = &shared_mem[blockDim.x];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_h * out_w;
    if (idx >= total_elements) return;

    int w_out = idx % out_w;
    idx /= out_w;
    int h_out = idx % out_h;
    idx /= out_h;
    int c_out = idx % out_channels;
    int b = idx / out_channels;  // local batch index

    int g = c_out / channels_per_group;
    int m = c_out % channels_per_group;

    // Pre-compute input and weight indices for the thread
    int weight_offset = (g * channels_per_group + m) * kernel_h * kernel_w;
    
    // Cooperatively load weights into shared memory
    if (threadIdx.x < kernel_h * kernel_w) {
        shared_weights[threadIdx.x] = weight[weight_offset + threadIdx.x];
    }
    __syncthreads();

    float sum = 0.0f;
    
    // Calculate input tile dimensions
    int tile_h = h_out * stride_h - padding_h;
    int tile_w = w_out * stride_w - padding_w;
    
    // Load input tile into shared memory cooperatively
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = tile_h + kh * dilation_h;
            int w_in = tile_w + kw * dilation_w;
            
            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
                int weight_idx = kh * kernel_w + kw;
                
                // Use shared memory for weights
                sum += input[input_idx] * shared_weights[weight_idx];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    int out_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
    output[out_idx] = sum;
}

// Forward function that overlaps computation with memory operations using CUDA streams
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    // Overlap computation across sub-batches using CUDA streams
    int num_streams = (batch_size < 4) ? batch_size : 4;
    int batch_per_stream = (batch_size + num_streams - 1) / num_streams;
    int threads = 256;

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < num_streams; ++i) {
        int b_start = i * batch_per_stream;
        if (b_start >= batch_size) break;
        int sub_batch = std::min(batch_per_stream, batch_size - b_start);

        // Calculate pointer offsets for the current sub-batch
        const float* x_sub = x_ptr + b_start * in_channels * in_h * in_w;
        float* out_sub = out_ptr + b_start * out_channels * out_h * out_w;

        int total_elements = sub_batch * out_channels * out_h * out_w;
        int blocks = (total_elements + threads - 1) / threads;

        depthwise_conv2d_kernel<<<blocks, threads, 0, streams[i]>>>(
            x_sub,
            weight.data_ptr<float>(),
            bias_ptr,
            out_sub,
            sub_batch,  // local batch size for this stream
            in_channels,
            in_h,
            in_w,
            out_channels,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            channels_per_group
        );
    }

    // Synchronize and destroy all streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise Conv2D forward with streams (CUDA)");
}
