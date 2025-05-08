#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Shared memory tile dimensions
#define TILE_SIZE 16
#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void transposed_conv3d_shared_kernel(
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
    extern __shared__ scalar_t shared_mem[];
    
    // Block handles a 2D tile of output elements
    const int tile_row = blockIdx.y * TILE_SIZE;
    const int tile_col = blockIdx.x * TILE_SIZE;
    const int thread_idx = threadIdx.x;
    
    // Calculate batch and channel indices
    const int n = blockIdx.z / out_channels;
    const int out_c = blockIdx.z % out_channels;
    const int group = out_c / (out_channels / groups);
    const int out_c_local = out_c % (out_channels / groups);
    const int in_channels_per_group = in_channels / groups;

    // Preload weights into shared memory
    scalar_t* shared_weights = shared_mem;
    const int weights_per_thread = (kT * kH * kW + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    #pragma unroll
    for (int i = 0; i < weights_per_thread; i++) {
        const int weight_idx = thread_idx * weights_per_thread + i;
        if (weight_idx < kT * kH * kW) {
            const int kd = weight_idx / (kH * kW);
            const int kh = (weight_idx / kW) % kH;
            const int kw = weight_idx % kW;
            
            for (int ic = 0; ic < in_channels_per_group; ic++) {
                const int weight_offset = ((ic * (out_channels / groups) + out_c_local) * kT + kd) * kH * kW + kh * kW + kw;
                shared_weights[ic * kT * kH * kW + weight_idx] = weight[weight_offset];
            }
        }
    }
    __syncthreads();

    // Process output elements
    #pragma unroll
    for (int th = 0; th < TILE_SIZE; th += blockDim.x / TILE_SIZE) {
        const int h = tile_row + th + (thread_idx % (blockDim.x / TILE_SIZE));
        if (h >= out_height) continue;

        #pragma unroll
        for (int tw = 0; tw < TILE_SIZE; tw++) {
            const int w = tile_col + tw;
            if (w >= out_width) continue;

            // Process each depth
            for (int d = 0; d < out_depth; d++) {
                scalar_t sum = 0.0f;

                // Loop over input channels for current group
                for (int ic = 0; ic < in_channels_per_group; ic++) {
                    const int input_channel = group * in_channels_per_group + ic;
                    
                    // Compute contribution from each kernel position
                    #pragma unroll
                    for (int kd = 0; kd < kT; kd++) {
                        const int d_in_tmp = d + pad_d - kd;
                        if (d_in_tmp % stride_d != 0) continue;
                        const int d_in = d_in_tmp / stride_d;
                        if (d_in < 0 || d_in >= in_depth) continue;

                        #pragma unroll
                        for (int kh = 0; kh < kH; kh++) {
                            const int h_in_tmp = h + pad_h - kh;
                            if (h_in_tmp % stride_h != 0) continue;
                            const int h_in = h_in_tmp / stride_h;
                            if (h_in < 0 || h_in >= in_height) continue;

                            #pragma unroll
                            for (int kw = 0; kw < kW; kw++) {
                                const int w_in_tmp = w + pad_w - kw;
                                if (w_in_tmp % stride_w != 0) continue;
                                const int w_in = w_in_tmp / stride_w;
                                if (w_in < 0 || w_in >= in_width) continue;

                                const int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * in_height + h_in) * in_width + w_in;
                                const int weight_idx = ic * kT * kH * kW + (kd * kH + kh) * kW + kw;
                                
                                sum += input[input_idx] * shared_weights[weight_idx];
                            }
                        }
                    }
                }

                if (bias != nullptr) {
                    sum += bias[out_c];
                }

                const int output_idx = (((n * out_channels + out_c) * out_depth + d) * out_height + h) * out_width + w;
                output[output_idx] = sum;
            }
        }
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
    const int out_channels = weight.size(1) * groups;
    const int kT = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    const int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    const int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    const int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    const dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        N * out_channels
    );
    const int threads = BLOCK_SIZE;
    const int shared_mem_size = (in_channels / groups) * kT * kH * kW * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_shared_kernel", ([&] {
        transposed_conv3d_shared_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "ConvTranspose3d forward with shared memory optimization",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}