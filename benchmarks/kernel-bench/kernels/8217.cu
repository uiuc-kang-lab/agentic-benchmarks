#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    extern __shared__ char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    
    // Calculate base indices for coalesced access
    const int out_pixels = out_height * out_width;
    const int total_pixels = batch_size * out_pixels;
    
    for (int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         pixel_idx < total_pixels; 
         pixel_idx += blockDim.x * gridDim.x) {
        
        const int b = pixel_idx / out_pixels;
        const int pixel_offset = pixel_idx % out_pixels;
        const int oh = pixel_offset / out_width;
        const int ow = pixel_offset % out_width;
        
        if (b >= batch_size) continue;
        
        const int out_channels_per_group = out_channels / groups;
        const int in_channels_per_group = in_channels / groups;
        
        // Process output channels in aligned chunks
        #pragma unroll 4
        for (int oc_base = 0; oc_base < out_channels; oc_base += warp_size) {
            const int oc = oc_base + lane_id;
            if (oc >= out_channels) continue;
            
            const int g = oc / out_channels_per_group;
            const int oc_group = oc % out_channels_per_group;
            const int ic_start = g * in_channels_per_group;
            
            scalar_t val = (bias != nullptr) ? bias[oc] : scalar_t(0);
            
            // Load frequently accessed weight values into shared memory
            const int weight_offset = oc_group * kernel_h * kernel_w;
            if (lane_id < kernel_h * kernel_w) {
                shared_input[warp_id * kernel_h * kernel_w + lane_id] = 
                    weight[ic_start * out_channels_per_group * kernel_h * kernel_w + weight_offset + lane_id];
            }
            __syncwarp();
            
            #pragma unroll
            for (int kh = 0; kh < kernel_h; ++kh) {
                const int h_in = (oh - kh * dilation + padding) / stride;
                if ((oh - kh * dilation + padding) % stride != 0) continue;
                if (h_in < 0 || h_in >= in_height) continue;
                
                #pragma unroll
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int w_in = (ow - kw * dilation + padding) / stride;
                    if ((ow - kw * dilation + padding) % stride != 0) continue;
                    if (w_in < 0 || w_in >= in_width) continue;
                    
                    const int kernel_idx = kh * kernel_w + kw;
                    
                    #pragma unroll 4
                    for (int ic = 0; ic < in_channels_per_group; ++ic) {
                        const scalar_t x_val = input[
                            ((b * in_channels + (ic_start + ic)) * in_height + h_in) * in_width + w_in
                        ];
                        
                        const scalar_t w_val = shared_input[warp_id * kernel_h * kernel_w + kernel_idx];
                        val += x_val * w_val;
                    }
                }
            }
            
            if (oc < out_channels) {
                output[((b * out_channels + oc) * out_height + oh) * out_width + ow] = val;
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation = 1
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->device().is_cuda(), "Bias must be a CUDA tensor");
    }

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int shared_memory_size = warps_per_block * kernel_h * kernel_w * sizeof(float);
    const int blocks = std::min(65535, (batch_size * out_height * out_width + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            groups,
            dilation,
            out_height,
            out_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 2D convolution (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}