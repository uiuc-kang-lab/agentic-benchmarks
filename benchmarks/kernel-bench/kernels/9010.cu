#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Shared memory size is determined dynamically based on kernel parameters
__global__ void conv1d_shared_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    
    // Shared memory layout: [input_chunk][weight_chunk]
    float* shared_input = shared_mem;
    float* shared_weight = &shared_mem[blockDim.x];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    // Calculate position indices
    int o = idx % out_size;
    int tmp = idx / out_size;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    // Initialize accumulator
    float sum = 0.0f;
    
    // Calculate start position for convolution window
    int start_pos = o * stride;
    int end_pos = start_pos + (kernel_size - 1) * dilation;
    
    // Process convolution in chunks to fit in shared memory
    for (int ic = 0; ic < in_channels; ++ic) {
        // Cooperatively load input data into shared memory
        if (start_pos < in_size) {
            for (int k = threadIdx.x; k < kernel_size && (start_pos + k * dilation) < in_size; k += blockDim.x) {
                int input_idx = b * (in_channels * in_size) + ic * in_size + start_pos + k * dilation;
                shared_input[k] = x[input_idx];
            }
        }
        
        // Cooperatively load weights into shared memory
        for (int k = threadIdx.x; k < kernel_size; k += blockDim.x) {
            int weight_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k;
            shared_weight[k] = weight[weight_idx];
        }
        
        // Ensure all data is loaded
        __syncthreads();
        
        // Compute partial convolution
        if (start_pos < in_size) {
            #pragma unroll
            for (int k = 0; k < kernel_size; ++k) {
                if (start_pos + k * dilation < in_size) {
                    sum += shared_input[k] * shared_weight[k];
                }
            }
        }
        
        // Synchronize before next iteration
        __syncthreads();
    }
    
    // Warp-level reduction for final sum
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Only the first thread in each warp writes the result
    if (threadIdx.x % warpSize == 0) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[b * (out_channels * out_size) + oc * out_size + o] = sum;
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
    const float* bias_data = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    int threads = 256;
    int blocks = (B * out_channels * out_size + threads - 1) / threads;
    
    // Calculate shared memory size
    int shared_mem_size = (kernel_size + threads) * sizeof(float);

    conv1d_shared_kernel<<<blocks, threads, shared_mem_size>>>(
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
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward with shared memory (CUDA)");
}