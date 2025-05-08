#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Custom CUDA kernel for transposed convolution
// This kernel computes a partial output for a batch chunk. Each thread processes one input element
// and accumulates its contribution over the kernel window to the corresponding output pixels.

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int chunkN,        // number of batches in this chunk
    int C_in, int H, int W,
    int C_out, int K,  // square kernel size
    int stride,
    int padding,
    int H_out, int W_out) {

    // total number of input elements in this batch chunk
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = chunkN * C_in * H * W;
    if (tid >= total) return;

    // Decode linear index to (n, c_in, h, w) within the chunk
    int w_idx = tid % W;
    int tmp = tid / W;
    int h_idx = tmp % H;
    tmp = tmp / H;
    int c_in = tmp % C_in;
    int n = tmp / C_in;  // local batch index within the chunk

    float in_val = input[tid];
    
    // Loop over kernel window
    for (int ki = 0; ki < K; ++ki) {
        for (int kj = 0; kj < K; ++kj) {
            int out_i = h_idx * stride - padding + ki;
            int out_j = w_idx * stride - padding + kj;
            if (out_i < 0 || out_i >= H_out || out_j < 0 || out_j >= W_out) continue;
            
            // Loop over all output channels
            for (int oc = 0; oc < C_out; ++oc) {
                // Weight shape is assumed to be [C_in, C_out, K, K]
                int weight_idx = c_in * (C_out * K * K) + oc * (K * K) + ki * K + kj;
                float w_val = weight[weight_idx];
                
                int out_index = n * (C_out * H_out * W_out) + oc * (H_out * W_out) + out_i * W_out + out_j;
                // Use atomic add to safely accumulate contributions
                atomicAdd(&output[out_index], in_val * w_val);
            }
        }
    }
}

// Kernel to add bias to each output channel
__global__ void add_bias_kernel(
    float* output,
    const float* bias,
    int total_elements,
    int C_out,
    int H_out,
    int W_out) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;
    // Compute channel index: output shape is [N, C_out, H_out, W_out]
    int channel = (tid / (H_out * W_out)) % C_out;
    output[tid] += bias[channel];
}

// Forward function implementing transposed convolution with CUDA streams
// This function partitions the input batch into chunks processed concurrently on separate streams.
// It overlaps kernel execution and memory operations, thus reducing runtime while maintaining full precision.

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Ensure tensors are on CUDA and are contiguous
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    if (bias.has_value()) {
        auto bias_val = bias.value();
        TORCH_CHECK(bias_val.is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias_val.is_contiguous(), "Bias tensor must be contiguous");
    }

    // Assume input tensor shape: [N, C_in, H, W]
    auto x_sizes = x.sizes();
    int N = x_sizes[0];
    int C_in = x_sizes[1];
    int H = x_sizes[2];
    int W = x_sizes[3];
    
    // Weight tensor shape assumed to be: [C_in, C_out, K, K] (square kernel)
    auto w_sizes = weight.sizes();
    int C_out = w_sizes[1];
    int K = w_sizes[2];

    // Compute output dimensions using the formula for ConvTranspose2d:
    // output_size = (input_size - 1) * stride - 2*padding + kernel_size + output_padding
    int H_out = (H - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W - 1) * stride - 2 * padding + K + output_padding;

    // Allocate the output tensor, initializing to zero
    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    // Use multiple CUDA streams to overlap computation with memory operations
    int nstreams = 2;
    std::vector<cudaStream_t> streams(nstreams);
    for (int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Partition the batch dimension among the available streams
    int chunk = (N + nstreams - 1) / nstreams;

    int block_size = 256;
    const float* weight_ptr = weight.data_ptr<float>();

    for (int i = 0; i < nstreams; i++) {
        int start = i * chunk;
        int end = std::min(N, (i + 1) * chunk);
        int chunkN = end - start;
        if (chunkN <= 0) continue;

        // Number of input elements for this batch chunk
        int num_elements = chunkN * C_in * H * W;
        const float* x_ptr = x.data_ptr<float>() + start * C_in * H * W;
        // Offset for the output pointer corresponding to this chunk
        float* out_ptr = output.data_ptr<float>() + start * C_out * H_out * W_out;

        int grid_size = (num_elements + block_size - 1) / block_size;

        // Launch the custom kernel asynchronously on the designated stream
        conv_transpose2d_kernel<<<grid_size, block_size, 0, streams[i]>>>(
            x_ptr,
            weight_ptr,
            out_ptr,
            chunkN, C_in, H, W,
            C_out, K,
            stride,
            padding,
            H_out, W_out
        );
    }

    // Synchronize and destroy the streams
    for (int i = 0; i < nstreams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // If a bias tensor is provided, add bias to the output
    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        int total_output = N * C_out * H_out * W_out;
        int block_bias = 256;
        int grid_bias = (total_output + block_bias - 1) / block_bias;
        add_bias_kernel<<<grid_bias, block_bias>>>(
            output.data_ptr<float>(),
            bias_tensor.data_ptr<float>(),
            total_output, C_out, H_out, W_out
        );
        cudaDeviceSynchronize();
    }

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward with streams (CUDA)");
}
