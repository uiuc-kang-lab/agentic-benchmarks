#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

__global__ void conv_transpose1d_kernel(
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
    extern __shared__ float shared_weights[];
    
    const int oc_block_size = blockDim.y;
    const int oc_block_idx = blockIdx.y * oc_block_size;
    const int oc = oc_block_idx + threadIdx.y;
    
    if (oc >= out_channels) return;

    // Load weights for this output channel block into shared memory
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int weight_idx = ic * out_channels * kernel_size + oc * kernel_size + k;
            shared_weights[threadIdx.y * in_channels * kernel_size + ic * kernel_size + k] = weight_ptr[weight_idx];
        }
    }
    __syncthreads();

    // Process spatial dimension
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.z;
    
    if (b >= batch_size || o >= output_length) return;

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int i_pos = o + padding - k * dilation;
        if (i_pos % stride != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = b * in_channels * input_length + ic * input_length + i;
            int weight_idx = threadIdx.y * in_channels * kernel_size + ic * kernel_size + k;
            sum += x_ptr[x_idx] * shared_weights[weight_idx];
        }
    }

    if (bias_ptr) sum += bias_ptr[oc];
    
    int output_idx = b * out_channels * output_length + oc * output_length + o;
    output_ptr[output_idx] = sum;
}

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");

    x = x.contiguous();
    weight = weight.contiguous();
    
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "Weight in_channels mismatch");
    
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        auto bias_contig = bias->contiguous();
        TORCH_CHECK(bias_contig.size(0) == out_channels, "Bias size mismatch");
        bias_ptr = bias_contig.data_ptr<float>();
    }

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    // Optimized block sizing: 128 threads in x (spatial), 4 in y (output channels)
    dim3 threads(128, 4);
    dim3 blocks(
        (output_length + threads.x - 1) / threads.x,
        (out_channels + threads.y - 1) / threads.y,
        batch_size
    );
    
    size_t shared_mem_size = threads.y * in_channels * kernel_size * sizeof(float);
    
    conv_transpose1d_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward_cuda, "ConvTranspose1D optimized with weight tiling",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}