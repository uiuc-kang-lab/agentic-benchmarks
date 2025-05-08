#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Custom CUDA kernel that computes contributions from each input element.
// Each thread processes one input element from a batch chunk and accumulates into the corresponding output region via atomic adds.
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int chunkN,       // number of batches in this chunk
    int C_in, int H, int W,
    int C_out, int K, // square kernel
    int stride,
    int padding,
    int H_out, int W_out) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = chunkN * C_in * H * W;
    if (tid >= total) return;

    // Decode the thread's linear index into (n, c_in, h, w) indices
    int w_idx = tid % W;
    int tmp = tid / W;
    int h_idx = tmp % H;
    tmp = tmp / H;
    int c_in = tmp % C_in;
    int n = tmp / C_in;  // local batch index within the chunk

    float in_val = input[tid];

    // For each kernel element, compute the corresponding output indices
    for (int ki = 0; ki < K; ++ki) {
        for (int kj = 0; kj < K; ++kj) {
            int out_i = h_idx * stride - padding + ki;
            int out_j = w_idx * stride - padding + kj;
            if (out_i < 0 || out_i >= H_out || out_j < 0 || out_j >= W_out) continue;

            // Loop over all output channels
            for (int oc = 0; oc < C_out; ++oc) {
                int weight_idx = c_in * (C_out * K * K) + oc * (K * K) + ki * K + kj;
                float w_val = weight[weight_idx];

                int out_index = n * (C_out * H_out * W_out) + oc * (H_out * W_out) + out_i * W_out + out_j;
                atomicAdd(&output[out_index], in_val * w_val);
            }
        }
    }
}

// Kernel to add bias if provided (each output channel gets its own bias value)
__global__ void add_bias_kernel(
    float* output,
    const float* bias,
    int total_elements,
    int C_out,
    int H_out,
    int W_out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    // Determine which output channel this element belongs to
    int channel = (tid / (H_out * W_out)) % C_out;
    output[tid] += bias[channel];
}

// Hybrid forward function that chooses between a built-in implementation and a custom kernel
// based on the input batch size. For small batches the built-in at::conv_transpose2d is preferred
// (as it may be optimized via cuDNN), while for large batches our custom kernel with multiple streams
// can yield improved GPU occupancy by overlapping work.

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Ensure inputs are CUDA tensors and contiguous
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    if (bias.has_value()) {
        auto bias_val = bias.value();
        TORCH_CHECK(bias_val.is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias_val.is_contiguous(), "Bias tensor must be contiguous");
    }

    // Retrieve input dimensions: x is [N, C_in, H, W]
    auto x_sizes = x.sizes();
    int N = x_sizes[0];
    int C_in = x_sizes[1];
    int H = x_sizes[2];
    int W = x_sizes[3];

    // Retrieve weight dimensions: weight is assumed to be [C_in, C_out, K, K]
    auto w_sizes = weight.sizes();
    int C_out = w_sizes[1];
    int K = w_sizes[2];

    // Compute output dimensions using the ConvTranspose2d formula
    int H_out = (H - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W - 1) * stride - 2 * padding + K + output_padding;

    // For small batches, the built-in implementation (often backed by cuDNN) performs very well
    if (N < 8) {  // threshold parameter; can be tuned
        return at::conv_transpose2d(
            x,
            weight,
            bias,
            {stride, stride},
            {padding, padding},
            {output_padding, output_padding},
            groups
        );
    }

    // Allocate output tensor and initialize to zero
    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    // Use multiple CUDA streams to partition the batch and overlap kernel execution
    int nstreams = 2;
    std::vector<cudaStream_t> streams(nstreams);
    for (int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Partition the batch dimension among streams
    int chunk = (N + nstreams - 1) / nstreams;
    int block_size = 256;
    const float* weight_ptr = weight.data_ptr<float>();

    for (int i = 0; i < nstreams; i++) {
        int start = i * chunk;
        int end = std::min(N, (i + 1) * chunk);
        int chunkN = end - start;
        if (chunkN <= 0) continue;

        // Number of input elements in this batch chunk
        int num_elements = chunkN * C_in * H * W;
        const float* x_ptr = x.data_ptr<float>() + start * C_in * H * W;
        // Output pointer offset corresponding to this batch chunk
        float* out_ptr = output.data_ptr<float>() + start * C_out * H_out * W_out;

        int grid_size = (num_elements + block_size - 1) / block_size;
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

    // Synchronize and destroy streams
    for (int i = 0; i < nstreams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // If bias is provided, launch a kernel to add it to the output
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
    m.def("forward", &conv_transpose2d_forward, "Hybrid ConvTranspose2d forward (CUDA)");
}
