#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// Initialize output with bias using 2D block configuration
__global__ void initialize_output_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int batch,
    const int out_channels,
    const int out_h,
    const int out_w) {
    
    // Use 2D blocks for better spatial locality
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;
    
    if (x >= out_w || y >= out_h || n >= batch) return;
    
    // Process all channels for this spatial location
    for (int oc = 0; oc < out_channels; oc++) {
        const int idx = n * (out_channels * out_h * out_w) +
                       oc * (out_h * out_w) +
                       y * out_w + x;
        output[idx] = bias[oc];
    }
}

// Scatter-based transposed convolution using 3D grid
__global__ void conv_transposed2d_scatter_atomic_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int in_h,
    const int in_w,
    const int out_channels_per_group,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int out_h,
    const int out_w,
    const int in_channels_per_group) {

    // 3D grid mapping: (spatial_x, channel, batch)
    const int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    const int n = blockIdx.z;
    
    if (c >= in_channels_per_group * groups) return;
    
    // Convert spatial_idx to ih, iw
    const int ih = spatial_idx / in_w;
    const int iw = spatial_idx % in_w;
    if (ih >= in_h) return;
    
    const int group = c / in_channels_per_group;
    const int c_within_group = c % in_channels_per_group;
    
    // Load input value once
    const int x_idx = n * (in_channels_per_group * groups * in_h * in_w) +
                     c * (in_h * in_w) +
                     ih * in_w + iw;
    const float x_val = x[x_idx];
    
    // Pre-compute base indices for weights
    const int weight_base = c_within_group * (out_channels_per_group * kernel_h * kernel_w);
    
    // Process output contributions
    #pragma unroll 4
    for (int kh = 0; kh < kernel_h; kh++) {
        const int oh = ih * stride_h - pad_h + kh * dilation_h;
        if (oh < 0 || oh >= out_h) continue;
        
        #pragma unroll 4
        for (int kw = 0; kw < kernel_w; kw++) {
            const int ow = iw * stride_w - pad_w + kw * dilation_w;
            if (ow < 0 || ow >= out_w) continue;
            
            const int khw_offset = kh * kernel_w + kw;
            
            // Process all output channels for this group
            #pragma unroll 4
            for (int oc_offset = 0; oc_offset < out_channels_per_group; oc_offset++) {
                const int weight_idx = weight_base +
                                     oc_offset * (kernel_h * kernel_w) +
                                     khw_offset;
                                     
                const float w_val = weight[weight_idx];
                const float contrib = x_val * w_val;
                
                if (contrib != 0.0f) {  // Skip atomic if contribution is zero
                    const int oc = group * out_channels_per_group + oc_offset;
                    const int out_idx = n * (groups * out_channels_per_group * out_h * out_w) +
                                      oc * (out_h * out_w) +
                                      oh * out_w + ow;
                    atomicAdd(&output[out_idx], contrib);
                }
            }
        }
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {
    
    x = x.contiguous();
    weight = weight.contiguous();
    
    if (!bias.has_value() || !bias.value().defined()) {
        bias = at::zeros({weight.size(1) * groups}, weight.options());
    } else {
        bias = bias.value().contiguous();
    }

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels_per_group = weight.size(1);
    const int out_channels = out_channels_per_group * groups;
    
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];
    
    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

    auto output = at::empty({batch, out_channels, out_h, out_w}, x.options());

    // Initialize output with bias using 2D blocks
    dim3 threads_init(16, 16);
    dim3 blocks_init(
        (out_w + threads_init.x - 1) / threads_init.x,
        (out_h + threads_init.y - 1) / threads_init.y,
        batch
    );
    
    initialize_output_kernel<<<blocks_init, threads_init>>>(
        output.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        batch,
        out_channels,
        out_h,
        out_w
    );

    // Launch scatter kernel with 3D grid
    const int THREADS_PER_BLOCK = 256;
    const int spatial_blocks = (in_h * in_w + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    dim3 grid(spatial_blocks, in_channels, batch);
    
    conv_transposed2d_scatter_atomic_kernel<<<grid, THREADS_PER_BLOCK>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        in_h,
        in_w,
        out_channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        out_h,
        out_w,
        in_channels / groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Transposed Convolution (CUDA)");
}