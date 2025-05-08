#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void transposed_conv3d_tiled_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int N, const int in_channels, const int in_depth, const int in_height, const int in_width,
    const int out_channels, const int out_depth, const int out_height, const int out_width,
    const int kT, const int kH, const int kW,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int out_pad_d, const int out_pad_h, const int out_pad_w,
    const int groups
) {
    __shared__ scalar_t shared_weight[TILE_SIZE][TILE_SIZE];
    
    // Calculate tile indices
    const int tile_idx = blockIdx.x;
    const int total_tiles = (N * out_channels * out_depth * out_height * ((out_width + TILE_SIZE - 1) / TILE_SIZE));
    
    if (tile_idx >= total_tiles) return;

    // Decompose tile index
    int remaining = tile_idx;
    const int tile_w = remaining % ((out_width + TILE_SIZE - 1) / TILE_SIZE);
    remaining /= ((out_width + TILE_SIZE - 1) / TILE_SIZE);
    const int h = remaining % out_height;
    remaining /= out_height;
    const int d = remaining % out_depth;
    remaining /= out_depth;
    const int c = remaining % out_channels;
    const int n = remaining / out_channels;

    // Thread index within the block
    const int thread_idx = threadIdx.x;
    
    // Calculate group and local channel index
    const int group = c / (out_channels / groups);
    const int out_c_local = c % (out_channels / groups);
    const int in_channels_per_group = in_channels / groups;

    // Calculate starting point for this tile
    const int w_start = tile_w * TILE_SIZE;
    const int w_end = min(w_start + TILE_SIZE, out_width);

    // Process elements within the tile
    for (int w = w_start + thread_idx; w < w_end; w += BLOCK_SIZE) {
        scalar_t sum = 0;

        // Loop over input channels for current group
        for (int ic = 0; ic < in_channels_per_group; ic++) {
            const int input_channel = group * in_channels_per_group + ic;
            
            // Load weight tile into shared memory
            if (thread_idx < TILE_SIZE) {
                for (int i = 0; i < TILE_SIZE; i += BLOCK_SIZE/TILE_SIZE) {
                    const int weight_idx = ((((input_channel) * (out_channels / groups) + out_c_local) * kT) * kH * kW) + thread_idx + i;
                    if (thread_idx + i < kT * kH * kW) {
                        shared_weight[thread_idx/TILE_SIZE][thread_idx%TILE_SIZE] = weight[weight_idx];
                    }
                }
            }
            __syncthreads();

            // Compute contribution from this input channel
            for (int kd = 0; kd < kT; kd++) {
                int d_in_tmp = d + pad_d - kd;
                if (d_in_tmp % stride_d != 0) continue;
                int d_in = d_in_tmp / stride_d;
                if (d_in < 0 || d_in >= in_depth) continue;

                for (int kh = 0; kh < kH; kh++) {
                    int h_in_tmp = h + pad_h - kh;
                    if (h_in_tmp % stride_h != 0) continue;
                    int h_in = h_in_tmp / stride_h;
                    if (h_in < 0 || h_in >= in_height) continue;

                    for (int kw = 0; kw < kW; kw++) {
                        int w_in_tmp = w + pad_w - kw;
                        if (w_in_tmp % stride_w != 0) continue;
                        int w_in = w_in_tmp / stride_w;
                        if (w_in < 0 || w_in >= in_width) continue;

                        const int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * in_height + h_in) * in_width + w_in;
                        const int weight_idx = (kd * kH + kh) * kW + kw;
                        
                        sum += input[input_idx] * shared_weight[weight_idx/TILE_SIZE][weight_idx%TILE_SIZE];
                    }
                }
            }
            __syncthreads();
        }

        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[c];
        }

        // Write output
        const int output_idx = (((n * out_channels + c) * out_depth + d) * out_height + h) * out_width + w;
        output[output_idx] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    input = input.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value().contiguous();
    }

    const int N = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    const int kT = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);
    const int out_channels = weight.size(1) * groups;

    const int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    const int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    const int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    const int total_tiles = N * out_channels * out_depth * out_height * ((out_width + TILE_SIZE - 1) / TILE_SIZE);
    const dim3 blocks(total_tiles);
    const dim3 threads(BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_tiled_kernel", ([&] {
        transposed_conv3d_tiled_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias_tensor.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, in_channels, in_depth, in_height, in_width,
            out_channels, out_depth, out_height, out_width,
            kT, kH, kW,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            output_padding[0], output_padding[1], output_padding[2],
            groups
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward with tiled processing",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}