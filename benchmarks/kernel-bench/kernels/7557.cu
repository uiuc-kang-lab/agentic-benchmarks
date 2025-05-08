#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void transposed_conv3d_tiled_coalesced_kernel(
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
    // Shared memory for weight caching
    extern __shared__ char shared_memory[];
    scalar_t* shared_weight = reinterpret_cast<scalar_t*>(shared_memory);
    
    // Thread indexing for coalesced memory access
    const int tid = threadIdx.x;
    const int warp_size = 32;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    
    // Block handles a slice of batch and output channels
    const int n_start = blockIdx.x;
    const int c_start = blockIdx.y * blockDim.y + warp_id;
    
    if (n_start >= N || c_start >= out_channels) return;

    // Each warp processes consecutive elements along width dimension
    for (int d = 0; d < out_depth; d++) {
        for (int h = 0; h < out_height; h++) {
            // Process multiple width positions per thread in a coalesced manner
            for (int w_base = 0; w_base < out_width; w_base += warp_size) {
                const int w = w_base + lane_id;
                if (w >= out_width) continue;

                scalar_t sum = 0;
                const int group = c_start / (out_channels / groups);
                const int out_c_local = c_start % (out_channels / groups);
                const int in_channels_per_group = in_channels / groups;

                // Load frequently accessed weights into shared memory
                if (lane_id < kW) {
                    for (int ic = 0; ic < in_channels_per_group; ic++) {
                        for (int kt = 0; kt < kT; kt++) {
                            for (int kh = 0; kh < kH; kh++) {
                                const int weight_idx = ((((group * in_channels_per_group + ic) * 
                                    (out_channels / groups) + out_c_local) * kT + kt) * kH + kh) * kW + lane_id;
                                shared_weight[((ic * kT + kt) * kH + kh) * kW + lane_id] = weight[weight_idx];
                            }
                        }
                    }
                }
                __syncthreads();

                // Compute output value with coalesced memory access
                for (int ic = 0; ic < in_channels_per_group; ic++) {
                    const int input_channel = group * in_channels_per_group + ic;
                    
                    for (int kt = 0; kt < kT; kt++) {
                        const int d_in_tmp = d + pad_d - kt;
                        if (d_in_tmp % stride_d != 0) continue;
                        const int d_in = d_in_tmp / stride_d;
                        if (d_in < 0 || d_in >= in_depth) continue;

                        for (int kh = 0; kh < kH; kh++) {
                            const int h_in_tmp = h + pad_h - kh;
                            if (h_in_tmp % stride_h != 0) continue;
                            const int h_in = h_in_tmp / stride_h;
                            if (h_in < 0 || h_in >= in_height) continue;

                            for (int kw = 0; kw < kW; kw++) {
                                const int w_in_tmp = w + pad_w - kw;
                                if (w_in_tmp % stride_w != 0) continue;
                                const int w_in = w_in_tmp / stride_w;
                                if (w_in < 0 || w_in >= in_width) continue;

                                // Coalesced input access
                                const int input_idx = (((n_start * in_channels + input_channel) * 
                                    in_depth + d_in) * in_height + h_in) * in_width + w_in;
                                
                                // Use cached weight
                                const scalar_t weight_val = shared_weight[((ic * kT + kt) * kH + kh) * kW + kw];
                                sum += input[input_idx] * weight_val;
                            }
                        }
                    }
                }

                if (bias != nullptr) {
                    sum += bias[c_start];
                }

                // Coalesced output write
                const int output_idx = (((n_start * out_channels + c_start) * 
                    out_depth + d) * out_height + h) * out_width + w;
                if (w < out_width) {
                    output[output_idx] = sum;
                }
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

    const int kT = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);
    const int out_channels = weight.size(1) * groups;

    const int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    const int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    const int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    // Configure block and grid dimensions for coalesced access
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * 32;
    const int out_channels_per_block = warps_per_block;
    
    dim3 blocks(N, (out_channels + out_channels_per_block - 1) / out_channels_per_block);
    
    // Shared memory size for weight caching
    const int shared_memory_size = (in_channels / groups) * kT * kH * kW * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_tiled_coalesced_kernel", ([&] {
        transposed_conv3d_tiled_coalesced_kernel<scalar_t><<<blocks, threads_per_block, shared_memory_size>>>(
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
    m.def("forward", &forward, "ConvTranspose3d forward with tiled coalesced memory access",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}