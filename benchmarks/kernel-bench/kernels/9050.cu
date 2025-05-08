#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv1d_shared_memory_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation
) {
    extern __shared__ float shared_weight[];
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int total_elements = B * out_channels * out_size;
    
    if (idx < total_elements) {
        const int o = idx % out_size;
        const int tmp = idx / out_size;
        const int oc = tmp % out_channels;
        const int b = tmp / out_channels;

        // Load weights into shared memory
        const int weights_per_thread = (kernel_size * in_channels + blockDim.x - 1) / blockDim.x;
        for (int i = 0; i < weights_per_thread; i++) {
            const int weight_idx = tid * weights_per_thread + i;
            if (weight_idx < kernel_size * in_channels) {
                shared_weight[weight_idx] = weight[oc * (in_channels * kernel_size) + weight_idx];
            }
        }
        __syncthreads();

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                const int input_pos = o * stride + k * dilation;
                if (input_pos < in_size) {
                    const int x_idx = b * (in_channels * in_size) + ic * in_size + input_pos;
                    const int w_idx = ic * kernel_size + k;
                    sum += x[x_idx] * shared_weight[w_idx];
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[oc];
        }

        const int out_idx = b * (out_channels * out_size) + oc * out_size + o;
        output[out_idx] = sum;
    }
}

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
    const float* bias_data = bias ? bias->data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    int threads = 256;
    int blocks = (B * out_channels * out_size + threads - 1) / threads;
    
    // Shared memory size for weights
    int shared_memory_size = in_channels * kernel_size * sizeof(float);

    conv1d_shared_memory_kernel<<<blocks, threads, shared_memory_size>>>(
        x_data,
        weight_data,
        bias_data,
        output_data,
        B,
        in_channels,
        in_size,
        out_channels,
        kernel_size,
        out_size,
        stride,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward (CUDA)");
}