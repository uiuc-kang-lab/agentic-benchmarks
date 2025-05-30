#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv1d_coalesced_kernel(
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
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + blockDim.x * kernel_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    int o = idx % out_size;
    idx /= out_size;
    int oc = idx % out_channels;
    int b = idx / out_channels;

    float sum = 0.0f;

    // Process input channels in tiles to maximize shared memory usage
    const int TILE_SIZE = 32;  // Process 32 input channels at a time
    for (int ic_base = 0; ic_base < in_channels; ic_base += TILE_SIZE) {
        int ic_end = min(ic_base + TILE_SIZE, in_channels);
        
        // Cooperatively load input data into shared memory
        for (int k = 0; k < kernel_size; k++) {
            int input_pos = o * stride + k * dilation;
            if (input_pos < in_size) {
                for (int ic_offset = threadIdx.x; ic_offset < (ic_end - ic_base); ic_offset += blockDim.x) {
                    int ic = ic_base + ic_offset;
                    int x_idx = b * in_channels * in_size + ic * in_size + input_pos;
                    shared_input[ic_offset * kernel_size + k] = __ldg(&x[x_idx]);
                }
            }
        }

        // Cooperatively load weights into shared memory
        for (int ic_offset = threadIdx.x; ic_offset < (ic_end - ic_base); ic_offset += blockDim.x) {
            int ic = ic_base + ic_offset;
            for (int k = 0; k < kernel_size; k++) {
                int w_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                shared_weight[ic_offset * kernel_size + k] = __ldg(&weight[w_idx]);
            }
        }

        __syncthreads();

        // Compute convolution using shared memory
        for (int ic = 0; ic < (ic_end - ic_base); ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                int input_pos = o * stride + k * dilation;
                if (input_pos < in_size) {
                    sum += shared_input[ic * kernel_size + k] * 
                           shared_weight[ic * kernel_size + k];
                }
            }
        }

        __syncthreads();
    }

    if (bias != nullptr) {
        sum += __ldg(&bias[oc]);
    }

    int out_idx = b * out_channels * out_size + oc * out_size + o;
    output[out_idx] = sum;
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

    conv1d_coalesced_kernel<<<blocks, threads>>>(
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
