#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

__global__ void initialize_output_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int batch,
    const int out_channels,
    const int out_h,
    const int out_w) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = batch * out_channels * out_h * out_w;
    
    for (int idx = tid; idx < total; idx += stride) {
        const int oc = (idx / (out_h * out_w)) % out_channels;
        output[idx] = __ldg(&bias[oc]);
    }
}

__global__ void conv_transposed2d_warp_shfl_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
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

    // Get warp and lane information
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int warp_idx = blockIdx.x * warps_per_block + warp_id;
    
    // Calculate total number of warps and distribute work accordingly
    const int total_pixels = batch * in_channels * in_h * in_w;
    const int total_warps = gridDim.x * warps_per_block;
    
    // Process input pixels in strides of warp_size
    for (int pixel_base = warp_idx * warp_size; pixel_base < total_pixels; pixel_base += total_warps * warp_size) {
        int pixel_idx = pixel_base + lane_id;
        if (pixel_idx >= total_pixels) continue;

        // Decode input indices
        const int iw = pixel_idx % in_w;
        int tmp = pixel_idx / in_w;
        const int ih = tmp % in_h;
        tmp /= in_h;
        const int c = tmp % in_channels;
        const int n = tmp / in_channels;
        
        const float x_val = (pixel_idx < total_pixels) ? __ldg(&x[pixel_idx]) : 0.0f;
        const int group = c / in_channels_per_group;
        
        // Pre-compute output position bounds
        const int out_row_start = ih * stride_h - pad_h;
        const int out_col_start = iw * stride_w - pad_w;
        
        // Process kernel positions
        for (int kh = 0; kh < kernel_h; kh++) {
            const int out_row = out_row_start + kh * dilation_h;
            if (out_row < 0 || out_row >= out_h) continue;
            
            for (int kw = 0; kw < kernel_w; kw++) {
                const int out_col = out_col_start + kw * dilation_w;
                if (out_col < 0 || out_col >= out_w) continue;
                
                // Process output channels in chunks aligned to warp size
                for (int oc_chunk = 0; oc_chunk < out_channels_per_group; oc_chunk += warp_size) {
                    const int oc_offset = oc_chunk + lane_id;
                    
                    float contrib = 0.0f;
                    if (oc_offset < out_channels_per_group && pixel_idx < total_pixels) {
                        const int weight_idx = c * (out_channels_per_group * kernel_h * kernel_w) +
                                             oc_offset * (kernel_h * kernel_w) +
                                             kh * kernel_w + kw;
                        contrib = x_val * __ldg(&weight[weight_idx]);
                    }
                    
                    // Reduce contributions within the warp
                    #pragma unroll
                    for (int offset = warp_size/2; offset > 0; offset /= 2) {
                        contrib += __shfl_down_sync(0xffffffff, contrib, offset);
                    }
                    
                    // First lane performs atomic update
                    if (lane_id == 0 && oc_chunk < out_channels_per_group) {
                        const int oc = group * out_channels_per_group + oc_chunk;
                        const int out_idx = n * (groups * out_channels_per_group * out_h * out_w) +
                                          oc * (out_h * out_w) +
                                          out_row * out_w + out_col;
                        atomicAdd(&output[out_idx], contrib);
                    }
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

    // Initialize output with bias
    const int threads_init = 256;
    const int blocks_init = (batch * out_channels * out_h * out_w + threads_init - 1) / threads_init;
    initialize_output_kernel<<<blocks_init, threads_init>>>(
        output.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        batch,
        out_channels,
        out_h,
        out_w);

    // Launch main computation kernel
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int num_blocks = (batch * in_channels * in_h * in_w + threads_per_block - 1) / threads_per_block;
    
    conv_transposed2d_warp_shfl_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
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
        in_channels / groups);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Transposed Convolution with Warp-Level Primitives (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}