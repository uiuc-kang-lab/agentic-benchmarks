#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions for better cache utilization
#define TILE_SIZE 16
#define BLOCK_SIZE 256

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
    const int output_padding,
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    __shared__ scalar_t shared_weight[TILE_SIZE][TILE_SIZE];
    
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    
    // Block indices
    const int block_x = blockIdx.x * TILE_SIZE;
    const int block_y = blockIdx.y * TILE_SIZE;
    const int block_z = blockIdx.z;
    
    // Thread indices within block
    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    
    // Calculate batch and channel indices
    const int b = block_z / out_channels;
    const int oc = block_z % out_channels;
    
    if (b >= batch_size) return;
    
    const int g = oc / out_channels_per_group;
    const int oc_group = oc % out_channels_per_group;
    const int ic_start = g * in_channels_per_group;
    
    // Initialize accumulator
    scalar_t val = (bias != nullptr) ? bias[oc] : 0.0f;
    
    // Calculate output positions
    const int oh = block_y + ty;
    const int ow = block_x + tx;
    
    if (oh < out_height && ow < out_width) {
        // Pre-compute input and weight strides
        const int input_b_offset = b * in_channels * in_height * in_width;
        const int weight_oc_offset = oc_group * kernel_h * kernel_w;
        
        for (int ic = 0; ic < in_channels_per_group; ++ic) {
            const int ic_idx = ic_start + ic;
            const int input_c_offset = ic_idx * in_height * in_width;
            const int weight_ic_offset = ic_idx * out_channels_per_group * kernel_h * kernel_w;
            
            for (int kh = 0; kh < kernel_h; kh += TILE_SIZE) {
                for (int kw = 0; kw < kernel_w; kw += TILE_SIZE) {
                    // Load weight tile into shared memory
                    if (ty < TILE_SIZE && tx < TILE_SIZE && 
                        (kh + ty) < kernel_h && (kw + tx) < kernel_w) {
                        shared_weight[ty][tx] = weight[
                            weight_ic_offset + 
                            weight_oc_offset + 
                            ((kernel_h - 1 - (kh + ty)) * kernel_w) + 
                            (kernel_w - 1 - (kw + tx))
                        ];
                    }
                    __syncthreads();
                    
                    // Process the tile
                    for (int i = 0; i < TILE_SIZE && (kh + i) < kernel_h; ++i) {
                        int h_in_base = oh - (kh + i) * dilation + padding;
                        if (h_in_base % stride != 0) continue;
                        int h_in = h_in_base / stride;
                        if (h_in < 0 || h_in >= in_height) continue;
                        
                        for (int j = 0; j < TILE_SIZE && (kw + j) < kernel_w; ++j) {
                            int w_in_base = ow - (kw + j) * dilation + padding;
                            if (w_in_base % stride != 0) continue;
                            int w_in = w_in_base / stride;
                            if (w_in < 0 || w_in >= in_width) continue;
                            
                            val += input[input_b_offset + input_c_offset + h_in * in_width + w_in] * 
                                  shared_weight[i][j];
                        }
                    }
                    __syncthreads();
                }
            }
        }
        
        // Write output
        if (oh < out_height && ow < out_width) {
            const int out_idx = b * out_channels * out_height * out_width +
                              oc * out_height * out_width +
                              oh * out_width +
                              ow;
            output[out_idx] = val;
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

    // Calculate grid dimensions for tiled approach
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    dim3 threads(BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
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
            output_padding,
            groups,
            dilation,
            out_height,
            out_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 2D convolution with block tiling (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}