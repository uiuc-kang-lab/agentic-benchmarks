#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Optimized CUDA kernel for transposed convolution with minimized warp divergence
__global__ void conv_transpose2d_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int chunkN,        // number of batches in this chunk
    int C_in, int H, int W,
    int C_out, int K,  // square kernel size
    int stride,
    int padding,
    int H_out, int W_out) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = chunkN * C_in * H * W;
    if (tid >= total) return;

    int w_idx = tid % W;
    int tmp = tid / W;
    int h_idx = tmp % H;
    tmp = tmp / H;
    int c_in = tmp % C_in;
    int n = tmp / C_in;

    float in_val = input[tid];
    for (int ki = 0; ki < K; ++ki) {
        for (int kj = 0; kj < K; ++kj) {
            int out_i = h_idx * stride - padding + ki;
            int out_j = w_idx * stride - padding + kj;
            if (out_i < 0 || out_i >= H_out || out_j < 0 || out_j >= W_out) continue;

            for (int oc = 0; oc < C_out; ++oc) {
                int weight_idx = c_in * (C_out * K * K) + oc * (K * K) + ki * K + kj;
                float w_val = weight[weight_idx];

                int out_index = n * (C_out * H_out * W_out) + oc * (H_out * W_out) + out_i * W_out + out_j;
                atomicAdd(&output[out_index], in_val * w_val);
            }
        }
    }
}

// Optimized kernel to add bias, minimizing warp divergence
__global__ void add_bias_kernel_optimized(
    float* output,
    const float* bias,
    int total_elements,
    int C_out,
    int H_out,
    int W_out) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;
    int oc = (index / (H_out * W_out)) % C_out;
    output[index] += bias[oc];
}

// Forward function implementing optimized transposed convolution
// This function leverages optimized kernels and CUDA streams for efficient execution
torch::Tensor conv_transpose2d_forward_optimized(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    auto x_sizes = x.sizes();
    int N = x_sizes[0];
    int C_in = x_sizes[1];
    int H = x_sizes[2];
    int W = x_sizes[3];
    auto w_sizes = weight.sizes();
    int C_out = w_sizes[1];
    int K = w_sizes[2];

    int H_out = (H - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W - 1) * stride - 2 * padding + K + output_padding;

    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    int nstreams = 2;
    std::vector<cudaStream_t> streams(nstreams);
    for (int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int chunk = (N + nstreams - 1) / nstreams;
    int block_size = 256;
    const float* weight_ptr = weight.data_ptr<float>();

    for (int i = 0; i < nstreams; i++) {
        int start = i * chunk;
        int end = std::min(N, (i + 1) * chunk);
        int chunkN = end - start;
        if (chunkN <= 0) continue;

        int num_elements = chunkN * C_in * H * W;
        const float* x_ptr = x.data_ptr<float>() + start * C_in * H * W;
        float* out_ptr = output.data_ptr<float>() + start * C_out * H_out * W_out;

        int grid_size = (num_elements + block_size - 1) / block_size;

        conv_transpose2d_kernel_optimized<<<grid_size, block_size, 0, streams[i]>>>(
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

    for (int i = 0; i < nstreams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        int total_output = N * C_out * H_out * W_out;
        int block_bias = 256;
        int grid_bias = (total_output + block_bias - 1) / block_bias;
        add_bias_kernel_optimized<<<grid_bias, block_bias>>>(
            output.data_ptr<float>(),
            bias_tensor.data_ptr<float>(),
            total_output, C_out, H_out, W_out
        );
        cudaDeviceSynchronize();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_optimized", &conv_transpose2d_forward_optimized, "ConvTranspose2d forward optimized (CUDA)");
}
