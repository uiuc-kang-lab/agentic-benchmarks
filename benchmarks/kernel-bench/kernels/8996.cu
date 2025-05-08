#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(
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
    const unsigned int warp_size = 32;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    const unsigned int warps_per_block = blockDim.x / warp_size;
    const unsigned int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    
    // Calculate the total number of output elements that need to be computed
    const int total_warps = gridDim.x * warps_per_block;
    const int total_elements = B * out_channels * out_size;
    
    // Each warp processes multiple elements using stride pattern
    for (int idx = global_warp_id; idx < total_elements; idx += total_warps) {
        const int o = idx % out_size;
        const int tmp = idx / out_size;
        const int oc = tmp % out_channels;
        const int b = tmp / out_channels;

        // Initialize accumulator
        float sum = 0.0f;
        
        // Calculate input starting position
        const int start_pos = o * stride;
        
        // Each lane in the warp handles different input channels in parallel
        for (int ic = lane_id; ic < in_channels; ic += warp_size) {
            float lane_sum = 0.0f;
            
            // Compute partial convolution for this input channel
            for (int k = 0; k < kernel_size; k++) {
                const int input_pos = start_pos + k * dilation;
                if (input_pos < in_size) {
                    const int x_idx = b * (in_channels * in_size) + ic * in_size + input_pos;
                    const int w_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k;
                    lane_sum += x[x_idx] * weight[w_idx];
                }
            }
            
            // Warp-level reduction using shuffle operations
            #pragma unroll
            for (int offset = warp_size/2; offset > 0; offset /= 2) {
                lane_sum += __shfl_down_sync(0xffffffff, lane_sum, offset);
            }
            
            // First lane accumulates the result
            if (lane_id == 0) {
                sum += lane_sum;
            }
        }
        
        // Only the first lane writes the result
        if (lane_id == 0) {
            if (bias != nullptr) {
                sum += bias[oc];
            }
            const int out_idx = b * (out_channels * out_size) + oc * out_size + o;
            output[out_idx] = sum;
        }
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

    // Configure kernel launch parameters
    const int threads_per_block = 256; // Must be multiple of 32 (warp size)
    const int num_warps_per_block = threads_per_block / 32;
    const int num_blocks = (B * out_channels * out_size + num_warps_per_block - 1) / num_warps_per_block;

    conv1d_kernel<<<num_blocks, threads_per_block>>>(
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
    m.def("forward", &forward, "1D convolution forward (CUDA) with warp shuffle optimization");
}