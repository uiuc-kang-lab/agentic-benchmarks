#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Shared memory tile sizes
#define TILE_DIM 8
#define BLOCK_ROWS 8
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
    extern __shared__ char shared_memory[];
    scalar_t* shared_weights = reinterpret_cast<scalar_t*>(shared_memory);
    
    // Calculate thread and block indices
    const int tidx = threadIdx.x;
    const int batch_idx = blockIdx.z;
    const int out_ch_idx = blockIdx.y;
    const int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Early exit if outside bounds
    if (spatial_idx >= out_depth * out_height * out_width) return;
    
    // Calculate output coordinates
    const int out_w = spatial_idx % out_width;
    const int out_h = (spatial_idx / out_width) % out_height;
    const int out_d = spatial_idx / (out_width * out_height);
    
    // Calculate group and local channel indices
    const int group = out_ch_idx / (out_channels / groups);
    const int out_c_local = out_ch_idx % (out_channels / groups);
    const int in_channels_per_group = in_channels / groups;
    
    // Accumulator for output value
    scalar_t sum = 0.0f;
    
    // Load weights into shared memory in tiles
    const int weights_per_thread = (kT * kH * kW + BLOCK_SIZE - 1) / BLOCK_SIZE;
    #pragma unroll
    for (int load_idx = 0; load_idx < weights_per_thread; load_idx++) {
        const int weight_idx = tidx + load_idx * BLOCK_SIZE;
        if (weight_idx < kT * kH * kW) {
            const int kt = weight_idx / (kH * kW);
            const int kh = (weight_idx / kW) % kH;
            const int kw = weight_idx % kW;
            
            #pragma unroll
            for (int ic = 0; ic < in_channels_per_group; ic++) {
                const int weight_offset = ((ic * (out_channels / groups) + out_c_local) * kT * kH * kW) + weight_idx;
                shared_weights[ic * kT * kH * kW + weight_idx] = weight[weight_offset];
            }
        }
    }
    __syncthreads();
    
    // Process input elements
    #pragma unroll
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        const int input_channel = group * in_channels_per_group + ic;
        
        #pragma unroll
        for (int kt = 0; kt < kT; kt++) {
            const int d_in_tmp = out_d + pad_d - kt;
            if (d_in_tmp % stride_d != 0) continue;
            const int d_in = d_in_tmp / stride_d;
            if (d_in < 0 || d_in >= in_depth) continue;
            
            #pragma unroll
            for (int kh = 0; kh < kH; kh++) {
                const int h_in_tmp = out_h + pad_h - kh;
                if (h_in_tmp % stride_h != 0) continue;
                const int h_in = h_in_tmp / stride_h;
                if (h_in < 0 || h_in >= in_height) continue;
                
                #pragma unroll
                for (int kw = 0; kw < kW; kw++) {
                    const int w_in_tmp = out_w + pad_w - kw;
                    if (w_in_tmp % stride_w != 0) continue;
                    const int w_in = w_in_tmp / stride_w;
                    if (w_in < 0 || w_in >= in_width) continue;
                    
                    const int input_idx = ((batch_idx * in_channels + input_channel) * in_depth + d_in) * 
                                        in_height * in_width + h_in * in_width + w_in;
                    const int weight_idx = ic * kT * kH * kW + (kt * kH + kh) * kW + kw;
                    
                    sum += input[input_idx] * shared_weights[weight_idx];
                }
            }
        }
    }
    
    // Add bias if present
    if (bias != nullptr) {
        sum += bias[out_ch_idx];
    }
    
    // Write output
    const int out_idx = ((batch_idx * out_channels + out_ch_idx) * out_depth + out_d) *
                       out_height * out_width + out_h * out_width + out_w;
    output[out_idx] = sum;
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
    auto bias_tensor = bias.has_value() ? bias.value().contiguous() : torch::Tensor();
    
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
    
    const int threads = BLOCK_SIZE;
    const int spatial_blocks = (out_depth * out_height * out_width + threads - 1) / threads;
    
    const dim3 blocks(spatial_blocks, out_channels, N);
    
    // Calculate shared memory size
    const int shared_memory_size = (in_channels / groups) * kT * kH * kW * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_shared_kernel", ([&] {
        transposed_conv3d_shared_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
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