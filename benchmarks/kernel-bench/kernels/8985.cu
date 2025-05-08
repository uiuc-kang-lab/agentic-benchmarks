#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel that leverages shared memory to cache the weight filter for each (batch, output channel) pair
// Grid configuration: gridDim.x = B (batch size), gridDim.y = out_channels
// Each block loads the weight filter (of size in_channels * kernel_size) into shared memory
// and uses a grid-stride loop to compute the convolution over the output width (o dimension).

__global__ void conv1d_kernel_shared(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int in_channels,
    int in_size,
    int out_size,
    int kernel_size,
    int stride,
    int dilation
) {
    // Each block computes outputs for a fixed batch index (b) and output channel (oc)
    int b = blockIdx.x;
    int oc = blockIdx.y;

    // Allocate shared memory for the weight filter corresponding to this output channel
    extern __shared__ float sweight[];  // size = in_channels * kernel_size
    int filter_size = in_channels * kernel_size;

    // Load the weight filter from global memory into shared memory using a loop
    for (int i = threadIdx.x; i < filter_size; i += blockDim.x) {
        sweight[i] = weight[oc * filter_size + i];
    }
    __syncthreads();

    // Load the bias value for this output channel
    float bias_val = (bias != nullptr) ? bias[oc] : 0.0f;

    // Each thread computes one or more output positions along the width dimension
    for (int o = threadIdx.x; o < out_size; o += blockDim.x) {
        float sum = 0.0f;
        // Loop over all input channels and kernel positions
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                int input_pos = o * stride + k * dilation;
                if (input_pos < in_size) {
                    int x_index = b * (in_channels * in_size) + ic * in_size + input_pos;
                    sum += x[x_index] * sweight[ic * kernel_size + k];
                }
            }
        }
        sum += bias_val;
        // Compute the output index: output shape is (B, out_channels, out_size)
        int out_index = b * (gridDim.y * out_size) + oc * out_size + o;
        output[out_index] = sum;
    }
}

// Forward function exposed to PyTorch via pybind11
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels mismatch");

    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias->size(0) == weight.size(0), "Bias size mismatch");
    }

    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    // Grid configuration: one block per (batch, output channel) pair
    dim3 blocks(B, out_channels);
    int threads = 256;
    // Shared memory size for one weight filter
    int shared_mem_size = in_channels * kernel_size * sizeof(float);

    conv1d_kernel_shared<<<blocks, threads, shared_mem_size>>>(
        x_data,
        weight_data,
        bias_data,
        output_data,
        in_channels,
        in_size,
        out_size,
        kernel_size,
        stride,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward (CUDA) using shared memory for weights");
}
