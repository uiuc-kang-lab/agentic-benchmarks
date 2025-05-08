#include <torch/extension.h>

// Macros for input validation
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__global__ void transposed_conv3d_warp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_d, const int in_h, const int in_w,
    const int out_channels,
    const int out_d, const int out_h, const int out_w,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int groups,
    const int channels_per_group) {
    
    // Calculate thread position
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    // Calculate global position
    const int total_warps = (blockDim.x * gridDim.x) / WARP_SIZE;
    const int global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    
    // Total number of output elements
    const int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    
    // Process elements with stride of total warps
    for (int idx = global_warp_id; idx < total_elements; idx += total_warps) {
        // Decode 5D index
        const int w = idx % out_w;
        const int h = (idx / out_w) % out_h;
        const int d = (idx / (out_w * out_h)) % out_d;
        const int oc = (idx / (out_w * out_h * out_d)) % out_channels;
        const int n = idx / (out_w * out_h * out_d * out_channels);
        
        // Calculate group
        const int group = oc / channels_per_group;
        const int oc_within_group = oc % channels_per_group;
        
        // Initialize accumulator
        float sum = (bias != nullptr && lane_id == 0) ? bias[oc] : 0.0f;
        
        // Compute input bounds for this output position
        const int in_d_start = (d + pad_d - kernel_d + 1 + stride_d - 1) / stride_d;
        const int in_h_start = (h + pad_h - kernel_h + 1 + stride_h - 1) / stride_h;
        const int in_w_start = (w + pad_w - kernel_w + 1 + stride_w - 1) / stride_w;
        
        // Each thread in warp processes different parts of input channels
        const int ic_per_thread = (channels_per_group + WARP_SIZE - 1) / WARP_SIZE;
        
        for (int ic_offset = 0; ic_offset < ic_per_thread; ic_offset++) {
            const int ic_idx = lane_id + ic_offset * WARP_SIZE;
            if (ic_idx >= channels_per_group) continue;
            
            const int ic = group * channels_per_group + ic_idx;
            
            for (int kd = 0; kd < kernel_d; kd++) {
                const int in_d_idx = in_d_start + kd;
                if (in_d_idx < 0 || in_d_idx >= in_d) continue;
                
                for (int kh = 0; kh < kernel_h; kh++) {
                    const int in_h_idx = in_h_start + kh;
                    if (in_h_idx < 0 || in_h_idx >= in_h) continue;
                    
                    for (int kw = 0; kw < kernel_w; kw++) {
                        const int in_w_idx = in_w_start + kw;
                        if (in_w_idx < 0 || in_w_idx >= in_w) continue;
                        
                        // Load input value
                        const int in_idx = ((n * in_channels + ic) * in_d + in_d_idx) * in_h * in_w +
                                         in_h_idx * in_w + in_w_idx;
                        const float in_val = input[in_idx];
                        
                        // Load weight value
                        const int weight_idx = ((ic * channels_per_group + oc_within_group) * kernel_d + kd) *
                                             kernel_h * kernel_w + kh * kernel_w + kw;
                        const float weight_val = weight[weight_idx];
                        
                        // Accumulate partial sum
                        sum += in_val * weight_val;
                    }
                }
            }
        }
        
        // Warp-level reduction using shuffle operations
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        }
        
        // First thread in warp writes result
        if (lane_id == 0) {
            output[idx] = sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(*bias);
    }
    
    auto input_sizes = x.sizes();
    auto weight_sizes = weight.sizes();
    
    const int batch_size = input_sizes[0];
    const int in_channels = input_sizes[1];
    const int in_d = input_sizes[2];
    const int in_h = input_sizes[3];
    const int in_w = input_sizes[4];
    
    const int out_channels = weight_sizes[1] * groups;
    const int kernel_d = weight_sizes[2];
    const int kernel_h = weight_sizes[3];
    const int kernel_w = weight_sizes[4];
    
    const int out_d = (in_d - 1) * stride[0] - 2 * padding[0] + kernel_d + output_padding[0];
    const int out_h = (in_h - 1) * stride[1] - 2 * padding[1] + kernel_h + output_padding[1];
    const int out_w = (in_w - 1) * stride[2] - 2 * padding[2] + kernel_w + output_padding[2];
    
    auto output = torch::zeros({batch_size, out_channels, out_d, out_h, out_w}, x.options());
    
    const int threads_per_block = 256;
    const int num_blocks = (batch_size * out_channels * out_d * out_h * out_w + threads_per_block - 1) / threads_per_block;
    
    transposed_conv3d_warp_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_d, in_h, in_w,
        out_channels,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        groups,
        in_channels / groups
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed Conv3D forward with warp primitives (CUDA)");
}