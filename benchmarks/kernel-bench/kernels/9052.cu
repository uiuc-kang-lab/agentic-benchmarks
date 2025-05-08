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
    extern __shared__ float shared_sum[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    int o = idx % out_size;
    idx /= out_size;
    int oc = idx % out_channels;
    int b = idx / out_channels;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = o * stride - k * dilation;
            if (input_pos >= 0 && input_pos < in_size) {
                int x_idx = b * (in_channels * in_size) + ic * in_size + input_pos;
                int w_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k;
                sum += __ldg(&x[x_idx]) * __ldg(&weight[w_idx]);
            }
        }
    }

    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    // Reduce within block using warp-level primitives
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        shared_sum[threadIdx.x / warpSize] = sum;
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x / warpSize) {
        sum = shared_sum[threadIdx.x];
        for (int offset = blockDim.x / (2 * warpSize); offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    if (threadIdx.x == 0) {
        output[b * (out_channels * out_size) + oc * out_size + o] = sum + (bias ? __ldg(&bias[oc]) : 0.0f);
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

    int total_elements = B * out_channels * out_size;
    int threads = 256; // Using a block size of 256, which is a multiple of warp size
    int blocks = (total_elements + threads - 1) / threads;

    conv1d_shared_memory_kernel<<<blocks, threads, threads * sizeof(float)>>>(
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
