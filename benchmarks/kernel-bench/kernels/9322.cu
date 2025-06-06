#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute output length given convolution parameters
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// Kernel using shared memory for weights and tuned block size
__global__ void conv_transpose1d_kernel_shared(
    const float* __restrict__ x_ptr,
    const float* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ output_ptr,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // grid configuration: blockIdx.y = batch index, blockIdx.x = output channel index
    int b = blockIdx.y;
    int oc = blockIdx.x;

    // Allocate shared memory for the weight slice for this output channel
    extern __shared__ float sweight[]; // size: in_channels * kernel_size floats
    int weight_count = in_channels * kernel_size;
    
    // Each thread loads portions of the weight into shared memory
    for (int idx = threadIdx.x; idx < weight_count; idx += blockDim.x) {
        int ic = idx / kernel_size;
        int k = idx % kernel_size;
        // Weight layout: [in_channels, out_channels, kernel_size]
        sweight[idx] = weight_ptr[ic * out_channels * kernel_size + oc * kernel_size + k];
    }
    __syncthreads();

    // Each thread computes one or more output positions along the temporal dimension
    for (int o = threadIdx.x; o < output_length; o += blockDim.x) {
        float sum = 0.0f;
        // Iterate over kernel positions
        for (int k = 0; k < kernel_size; ++k) {
            int i_pos = o + padding - k * dilation;
            if (i_pos % stride != 0) continue;
            int i = i_pos / stride;
            if (i < 0 || i >= input_length) continue;
            // Sum over all input channels
            for (int ic = 0; ic < in_channels; ++ic) {
                float w = sweight[ic * kernel_size + k];
                int x_idx = b * in_channels * input_length + ic * input_length + i;
                sum += x_ptr[x_idx] * w;
            }
        }
        if (bias_ptr) {
            sum += bias_ptr[oc];
        }
        int out_idx = b * out_channels * output_length + oc * output_length + o;
        output_ptr[out_idx] = sum;
    }
}

// Forward CUDA function with tuned block size based on runtime parameters
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

    // Tuned block size selection based on output_length for optimal occupancy on H100
    int threads_per_block;
    if (output_length <= 32) {
        threads_per_block = 32;
    } else if (output_length <= 64) {
        threads_per_block = 64;
    } else if (output_length <= 128) {
        threads_per_block = 128;
    } else if (output_length <= 256) {
        threads_per_block = 256;
    } else {
        threads_per_block = 512;
    }

    // Grid configuration: each block handles one (batch, out_channel) pair
    dim3 grid(out_channels, batch_size);
    size_t shared_mem_size = in_channels * kernel_size * sizeof(float);

    conv_transpose1d_kernel_shared<<<grid, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
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
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA) with tuned block size",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
