#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile sizes for shared memory optimization
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
    extern __shared__ char shared_mem[];
    
    // Shared memory for weight tiles and input tiles
    scalar_t* s_weight = (scalar_t*)shared_mem;
    scalar_t* s_input = (scalar_t*)(s_weight + TILE_SIZE * TILE_SIZE);

    const int tid = threadIdx.x;
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int idx = blockIdx.x * blockDim.x + tid;
    
    if (idx >= total_elements) return;

    // Calculate output indices
    int n = idx;
    const int ow = n % out_width;
    n /= out_width;
    const int oh = n % out_height;
    n /= out_height;
    const int oc = n % out_channels;
    n /= out_channels;
    const int b = n;

    // Calculate group information
    const int out_channels_per_group = out_channels / groups;
    const int g = oc / out_channels_per_group;
    const int oc_group = oc % out_channels_per_group;
    const int in_channels_per_group = in_channels / groups;
    const int ic_start = g * in_channels_per_group;

    // Initialize output value
    scalar_t val = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

    // Process input in tiles
    for (int tile_start = 0; tile_start < in_channels_per_group; tile_start += TILE_SIZE) {
        const int tile_end = min(tile_start + TILE_SIZE, in_channels_per_group);
        
        // Load weight tile into shared memory
        for (int i = tid; i < TILE_SIZE * kernel_h * kernel_w; i += BLOCK_SIZE) {
            const int ic_offset = i / (kernel_h * kernel_w);
            const int k_idx = i % (kernel_h * kernel_w);
            if (tile_start + ic_offset < tile_end) {
                s_weight[i] = weight[
                    (ic_start + tile_start + ic_offset) * (out_channels_per_group * kernel_h * kernel_w) +
                    oc_group * kernel_h * kernel_w +
                    k_idx
                ];
            }
        }
        
        __syncthreads();

        // Process the tile
        for (int ic = 0; ic < (tile_end - tile_start); ++ic) {
            // Load input tile into shared memory
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int h_in_base = oh - kh * dilation + padding;
                    int w_in_base = ow - kw * dilation + padding;
                    
                    if (h_in_base % stride == 0 && w_in_base % stride == 0) {
                        int h_in = h_in_base / stride;
                        int w_in = w_in_base / stride;
                        
                        if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                            const int input_idx = b * in_channels * in_height * in_width +
                                                (ic_start + tile_start + ic) * in_height * in_width +
                                                h_in * in_width + w_in;
                            const int shared_idx = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                            s_input[shared_idx] = input[input_idx];
                        } else {
                            s_input[ic * kernel_h * kernel_w + kh * kernel_w + kw] = 0;
                        }
                    }
                }
            }
            
            // Compute partial sum using shared memory
            for (int k = 0; k < kernel_h * kernel_w; ++k) {
                val += s_input[ic * kernel_h * kernel_w + k] *
                       s_weight[ic * kernel_h * kernel_w + k];
            }
        }
        
        __syncthreads();
    }

    output[idx] = val;
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

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    const int total_elements = output.numel();
    const int threads = BLOCK_SIZE;
    const int blocks = (total_elements + threads - 1) / threads;

    // Calculate shared memory size
    const size_t shared_mem_size = sizeof(float) * (
        TILE_SIZE * TILE_SIZE +  // For weights
        TILE_SIZE * kernel_h * kernel_w  // For input
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "Transposed 2D convolution with shared memory tiling (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}