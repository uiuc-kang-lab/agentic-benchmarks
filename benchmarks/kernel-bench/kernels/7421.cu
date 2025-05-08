#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized transposed convolution kernel using shared memory for weight caching
// to reduce global memory accesses and minimize synchronization overhead.

// This kernel loads the entire weight tensor into shared memory with one __syncthreads() call
// and then each thread computes one output element by gathering contributions from the input over
// all channels and kernel positions. Bias is added directly if provided.

__global__ void conv_transpose2d_kernel_shmem(
    const float* __restrict__ input,    // [N, C_in, H_in, W_in]
    const float* __restrict__ weight,   // [C_in, C_out, K, K]
    const float* __restrict__ bias,     // [C_out] or nullptr
    float* __restrict__ output,         // [N, C_out, H_out, W_out]
    int N,
    int C_in, int H_in, int W_in,
    int C_out,
    int K,
    int stride,
    int padding,
    int H_out, int W_out) {

    // Allocate shared memory for the weight tensor
    extern __shared__ float sweight[];  // size: C_in * C_out * K * K
    int weight_size = C_in * C_out * K * K;
    
    // Each thread in the block loads part of the weight into shared memory
    for (int i = threadIdx.x; i < weight_size; i += blockDim.x) {
        sweight[i] = weight[i];
    }
    // Synchronize to ensure complete weight load
    __syncthreads();

    // Compute global output index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (index >= total) return;

    // Decode linear index into (n, oc, h, w)
    int w_out_idx = index % W_out;
    int tmp = index / W_out;
    int h_out_idx = tmp % H_out;
    tmp /= H_out;
    int oc = tmp % C_out;
    int n = tmp / C_out;

    float sum = 0.0f;
    
    // Loop over input channels and kernel spatial dimensions
    for (int c = 0; c < C_in; c++) {
        for (int ki = 0; ki < K; ki++) {
            for (int kj = 0; kj < K; kj++) {
                // Compute the corresponding input coordinates
                int in_h = h_out_idx + padding - ki;
                int in_w = w_out_idx + padding - kj;
                // Check if the computed coordinate aligns with the stride
                if ((in_h % stride) != 0 || (in_w % stride) != 0) continue;
                in_h /= stride;
                in_w /= stride;
                if (in_h < 0 || in_h >= H_in || in_w < 0 || in_w >= W_in) continue;
                
                int input_idx = n * (C_in * H_in * W_in) + c * (H_in * W_in) + in_h * W_in + in_w;
                int weight_idx = c * (C_out * K * K) + oc * (K * K) + ki * K + kj;
                sum += input[input_idx] * sweight[weight_idx];
            }
        }
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    output[index] = sum;
}

// Forward function for transposed convolution using the optimized kernel
// It sets up the output dimensions and launches the CUDA kernel with dynamic shared memory allocation.

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

    // x shape: [N, C_in, H_in, W_in]
    auto x_sizes = x.sizes();
    int N = x_sizes[0];
    int C_in = x_sizes[1];
    int H_in = x_sizes[2];
    int W_in = x_sizes[3];

    // weight shape: [C_in, C_out, K, K]
    auto w_sizes = weight.sizes();
    int C_out = w_sizes[1];
    int K = w_sizes[2];

    // Compute output dimensions based on ConvTranspose2d formula
    int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());

    int total_output = N * C_out * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_output + block_size - 1) / block_size;

    // Calculate the shared memory size for caching the entire weight tensor
    size_t shmem_size = (size_t)C_in * C_out * K * K * sizeof(float);

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv_transpose2d_kernel_shmem<<<grid_size, block_size, shmem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, K,
        stride, padding,
        H_out, W_out
    );
    
    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Optimized ConvTranspose2d with shared memory (CUDA)");
}
