#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// CUDA kernel leveraging shared memory for the weight slice of a group
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    // Compute per-group channel counts.
    int in_channels_per_group = C_in / groups;
    int out_channels_per_group = C_out / groups;
    int weight_slice_size = in_channels_per_group * out_channels_per_group * kernel_h * kernel_w;

    // Declare dynamic shared memory for storing one group's weight slice
    extern __shared__ float s_weight[];

    // Global thread index and total stride
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_idx = blockDim.x * gridDim.x;

    // Let the first thread in each block decide a "preferred" group whose weight will be loaded into shared memory
    __shared__ int shared_group;
    {
        int total = N * C_out * H_out * W_out;
        if (threadIdx.x == 0) {
            if (blockIdx.x * blockDim.x < total) {
                int idx0 = blockIdx.x * blockDim.x;
                int w_out0 = idx0 % W_out;
                int tmp0 = idx0 / W_out;
                int h_out0 = tmp0 % H_out;
                tmp0 = tmp0 / H_out;
                int c_out0 = tmp0 % C_out;
                shared_group = c_out0 / out_channels_per_group;
            } else {
                shared_group = -1;
            }
        }
    }
    __syncthreads();

    // Load the weight slice for shared_group into shared memory
    if (shared_group != -1) {
        const float* weight_group_ptr = weight + shared_group * weight_slice_size;
        for (int i = threadIdx.x; i < weight_slice_size; i += blockDim.x) {
            s_weight[i] = weight_group_ptr[i];
        }
    }
    __syncthreads();

    int total = N * C_out * H_out * W_out;
    while (idx < total) {
        // Decode flat index into (n, c_out, h_out, w_out)
        int w_out = idx % W_out;
        int tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp = tmp / H_out;
        int c_out = tmp % C_out;
        int n = tmp / C_out;

        int group_idx = c_out / out_channels_per_group;
        int input_channel_start = group_idx * in_channels_per_group;
        int input_channel_end = input_channel_start + in_channels_per_group;

        float sum_val = (bias != nullptr) ? bias[c_out] : 0.0f;

        // Use the shared memory weight slice if this thread's group matches the block's shared group
        bool use_shared = (group_idx == shared_group);
        const float* cur_weight = use_shared ? s_weight : (weight + group_idx * weight_slice_size);

        // Iterate over kernel spatial positions
        for (int k_y = 0; k_y < kernel_h; ++k_y) {
            for (int k_x = 0; k_x < kernel_w; ++k_x) {
                int h_in_possible = h_out + padding_h - k_y * dilation_h;
                int w_in_possible = w_out + padding_w - k_x * dilation_w;
                if (h_in_possible % stride_h != 0 || w_in_possible % stride_w != 0) continue;
                int h_in = h_in_possible / stride_h;
                int w_in = w_in_possible / stride_w;
                if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) continue;
                
                // Sum over the input channels for this group
                for (int c_in = input_channel_start; c_in < input_channel_end; ++c_in) {
                    int input_index = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int local_c_in = c_in - input_channel_start;
                    int local_c_out = c_out - group_idx * out_channels_per_group;
                    int weight_index = ((local_c_in * out_channels_per_group + local_c_out) * kernel_h + k_y) * kernel_w + k_x;
                    sum_val += input[input_index] * cur_weight[weight_index];
                }
            }
        }

        int output_index = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
        output[output_index] = sum_val;

        idx += stride_idx;
    }
}

// Wrapper function that sets up output dimensions, computes shared memory size, and launches the CUDA kernel
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int padding_h = padding[0];
    int padding_w = padding[1];
    int output_padding_h = output_padding[0]; // used for output dimension
    int output_padding_w = output_padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    int out_channels_per_group = weight.size(1);
    int C_out = out_channels_per_group * groups;

    // Compute output spatial dimensions for the transposed convolution
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const int threads = 256;
    int total_elements = N * C_out * H_out * W_out;
    int blocks = (total_elements + threads - 1) / threads;

    int in_channels_per_group = C_in / groups;
    int out_channels_per_group_calc = C_out / groups;
    int weight_slice_size = in_channels_per_group * out_channels_per_group_calc * kernel_h * kernel_w;
    int shared_mem_size = weight_slice_size * sizeof(float);

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    conv_transpose2d_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward with shared memory for weight (CUDA)");
}
