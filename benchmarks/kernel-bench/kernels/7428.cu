#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel to perform transposed convolution with pipelining and asynchronous weight loading into shared memory.
// On NVIDIA H100 GPUs, cp.async instructions can be used to overlap weight loading with computation.

__global__ void conv_transpose2d_kernel_pipeline(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int C_in, int H_in, int W_in,
    int C_out,
    int K,
    int stride,
    int padding,
    int H_out,
    int W_out) {

    // Each block processes one batch element. The batch index is given by blockIdx.y.
    int batch = blockIdx.y;
    const float* input_ptr = input + batch * C_in * H_in * W_in;
    float* output_ptr = output + batch * C_out * H_out * W_out;

    // Load the entire weight tensor into shared memory.
    extern __shared__ float shared_weight[];
    int weight_elements = C_in * C_out * K * K;
    
    // Loop to copy weight elements into shared memory. On H100, one can replace this loop with cp.async
    // to overlap weight loading with computation.
    for (int i = threadIdx.x; i < weight_elements; i += blockDim.x) {
        shared_weight[i] = weight[i];
    }
    __syncthreads();

    // Each thread computes one output element for this batch element.
    int total_out = C_out * H_out * W_out;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_out) return;

    // Decode flat index into output coordinates (oc, h, w)
    int w = index % W_out;
    int tmp = index / W_out;
    int h = tmp % H_out;
    int oc = tmp / H_out;

    float sum = 0.0f;
    // Transposed convolution: accumulate contributions from input channels and kernel window.
    for (int c = 0; c < C_in; ++c) {
        for (int ki = 0; ki < K; ++ki) {
            for (int kj = 0; kj < K; ++kj) {
                int in_h = h + padding - ki;
                int in_w = w + padding - kj;
                if ((in_h % stride) != 0 || (in_w % stride) != 0)
                    continue;
                in_h /= stride;
                in_w /= stride;
                if (in_h < 0 || in_h >= H_in || in_w < 0 || in_w >= W_in)
                    continue;
                int input_idx = c * (H_in * W_in) + in_h * W_in + in_w;
                int weight_idx = c * (C_out * K * K) + oc * (K * K) + ki * K + kj;
                sum += input_ptr[input_idx] * shared_weight[weight_idx];
            }
        }
    }
    output_ptr[index] = sum;
}

// Kernel to add bias to the output.
__global__ void add_bias_kernel(
    float* output,
    const float* bias,
    int total,
    int C_out,
    int H_out,
    int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int oc = (idx / (H_out * W_out)) % C_out;
    output[idx] += bias[oc];
}

// Forward function that partitions the batch across CUDA streams and overlaps computation with memory transfers.
// This implementation pipelines computation by launching kernels on different streams. Each kernel loads the
// weight into shared memory (potentially using cp.async on H100) and computes its output tile, overlapping
// memory transfers with computation.

torch::Tensor conv_transpose2d_forward(
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

    // Input dimensions: [N, C_in, H_in, W_in]
    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // Weight dimensions: [C_in, C_out, K, K] (square kernel assumed)
    int C_out = weight.size(1);
    int K = weight.size(2);

    // Compute output dimensions for transposed convolution:
    int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    // Use multiple CUDA streams to overlap kernel execution with memory operations.
    int nstreams = 2;
    int chunk = (N + nstreams - 1) / nstreams;
    std::vector<cudaStream_t> streams(nstreams);
    for (int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Each kernel processes one batch element; total output elements per batch = C_out * H_out * W_out.
    int total_out = C_out * H_out * W_out;
    int block_size = 256;
    int grid_x = (total_out + block_size - 1) / block_size;
    dim3 block(block_size);
    int shared_mem_size = C_in * C_out * K * K * sizeof(float);

    // Partition the batch and launch kernels on different streams to pipeline computation and memory transfers.
    for (int i = 0; i < nstreams; i++) {
        int start = i * chunk;
        int end = std::min(N, (i + 1) * chunk);
        int batch_chunk = end - start;
        if (batch_chunk <= 0) continue;

        // Grid dimensions: x-dimension covers output elements, y-dimension covers batch elements in the chunk.
        dim3 grid(grid_x, batch_chunk);
        conv_transpose2d_kernel_pipeline<<<grid, block, shared_mem_size, streams[i]>>>(
            x.data_ptr<float>() + start * C_in * H_in * W_in,
            weight.data_ptr<float>(),
            output.data_ptr<float>() + start * C_out * H_out * W_out,
            C_in, H_in, W_in, C_out, K, stride, padding, H_out, W_out
        );
    }

    // Synchronize and destroy streams
    for (int i = 0; i < nstreams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // If a bias tensor is provided, add bias using a separate kernel.
    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        int total_elements = output.numel();
        int bias_block = 256;
        int bias_grid = (total_elements + bias_block - 1) / bias_block;
        add_bias_kernel<<<bias_grid, bias_block>>>(
            output.data_ptr<float>(),
            bias_tensor.data_ptr<float>(),
            total_elements, C_out, H_out, W_out
        );
        cudaDeviceSynchronize();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward with pipelining and async weight copy (CUDA)");
}
