#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Helper function to compute the output length
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// Kernel that uses shared memory to load the weight tile for a fixed output channel
// Each block is mapped to a unique (batch, out_channel) pair. Threads in the block then compute
// the output values along the output length dimension.
// __syncthreads() is used only once after loading the shared memory tile to ensure consistency.

__global__ void conv_transpose1d_shared_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ output_ptr,
    int in_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Map each block to a unique (batch, output channel) pair
    int b = blockIdx.x;      // batch index
    int oc = blockIdx.y;     // output channel index

    // Allocate shared memory for the weight tile: [in_channels x kernel_size]
    extern __shared__ float s_weight[];
    int tile_size = in_channels * kernel_size;

    // Each thread loads part of the weight tile from global memory into shared memory
    // Global weight shape: [in_channels, out_channels, kernel_size]
    // For the given oc, the element weight[ic, oc, k] is at index: ic * (out_channels * kernel_size) + oc * kernel_size + k
    for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
        int ic = idx / kernel_size;
        int k = idx % kernel_size;
        int weight_idx = ic * (gridDim.y * kernel_size) + oc * kernel_size + k;
        s_weight[idx] = weight_ptr[weight_idx];
    }
    __syncthreads(); // Ensure the shared weight tile is fully loaded before use

    // Each thread processes output elements along the output_length dimension
    for (int o = threadIdx.x; o < output_length; o += blockDim.x) {
        float sum = 0.0f;
        // Loop over kernel positions
        for (int k = 0; k < kernel_size; ++k) {
            int i_pos = o + padding - k * dilation;
            if (i_pos % stride != 0) continue;
            int i = i_pos / stride;
            if (i < 0 || i >= input_length) continue;
            // Accumulate over input channels
            for (int ic = 0; ic < in_channels; ++ic) {
                int x_idx = b * (in_channels * input_length) + ic * input_length + i;
                sum += x_ptr[x_idx] * s_weight[ic * kernel_size + k];
            }
        }
        if (bias_ptr) {
            sum += bias_ptr[oc];
        }
        int out_idx = b * (gridDim.y * output_length) + oc * output_length + o;
        output_ptr[out_idx] = sum;
    }
}

// Forward function for the CUDA kernel
torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, in_channels, input_length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");

    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_contig;
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_contig = bias->contiguous();
        TORCH_CHECK(bias_contig.device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_contig.dim() == 1, "bias must be 1D");
        bias_ptr = bias_contig.data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight's in_channels must match x's in_channels");
    if (bias.has_value()) {
        TORCH_CHECK(bias_contig.size(0) == out_channels, "bias size must match out_channels");
    }

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    // Configure grid: each block computes one (batch, out_channel) pair
    dim3 grid(batch_size, out_channels);
    int threads_per_block = 256;
    // Shared memory size needed per block: in_channels * kernel_size * sizeof(float)
    size_t shared_mem_size = in_channels * kernel_size * sizeof(float);

    conv_transpose1d_shared_kernel<<<grid, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        in_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA) with shared memory optimization",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
