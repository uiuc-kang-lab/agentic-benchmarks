#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <int BLOCK_SIZE>
__global__ void conv1d_shared_warp_kernel(
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
    float* shared_x = shared_mem;
    float* shared_weight = shared_mem + in_size;

    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    int o = idx % out_size;
    idx /= out_size;
    int oc = idx % out_channels;
    int b = idx / out_channels;

    float sum = 0.0f;

    // Load weights into shared memory
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = threadIdx.x; k < kernel_size; k += BLOCK_SIZE) {
            int w_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k;
            shared_weight[ic * kernel_size + k] = weight[w_idx];
        }
    }

    // Load input data into shared memory
    int input_start = o * stride;
    for (int k = threadIdx.x; k < kernel_size * dilation; k += BLOCK_SIZE) {
        int input_pos = input_start + k;
        if (input_pos < in_size) {
            shared_x[input_pos] = x[b * (in_channels * in_size) + input_pos];
        }
    }
    
    __syncthreads();

    // Compute convolution using shared memory
    for (int ic = 0; ic < in_channels; ++ic) {
        float warp_sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = o * stride + k * dilation;
            if (input_pos < in_size) {
                warp_sum += shared_x[input_pos] * shared_weight[ic * kernel_size + k];
            }
        }
        
        // Warp-level reduction
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (lane_id == 0) {
            sum += warp_sum;
        }
    }

    if (bias != nullptr && lane_id == 0) {
        sum += bias[oc];
    }
    
    // Write output
    if (lane_id == 0) {
        int out_idx = b * (out_channels * out_size) + oc * out_size + o;
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

    constexpr int BLOCK_SIZE = 256;
    int total_elements = B * out_channels * out_size;
    int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Calculate shared memory size
    size_t shared_mem_size = (in_size + in_channels * kernel_size) * sizeof(float);

    conv1d_shared_warp_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, shared_mem_size>>>(
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