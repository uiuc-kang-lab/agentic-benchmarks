#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4
#define CHUNK_SIZE 32

template <typename scalar_t>
__global__ void transposed_conv3d_stream_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int out_depth, int out_height, int out_width,
    int kT, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups,
    int batch_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread for better utilization
    for (int i = idx; i < out_channels * out_depth * out_height * out_width; i += stride) {
        int w = i % out_width;
        int h = (i / out_width) % out_height;
        int d = (i / (out_width * out_height)) % out_depth;
        int c = (i / (out_width * out_height * out_depth));
        
        // Add batch_offset to process the correct batch
        int n = batch_offset;
        
        int group = c / (out_channels / groups);
        int out_c_local = c % (out_channels / groups);
        
        scalar_t sum = 0;
        
        #pragma unroll 4
        for (int ic = 0; ic < in_channels / groups; ic++) {
            int input_channel = group * (in_channels / groups) + ic;
            
            #pragma unroll 2
            for (int kd = 0; kd < kT; kd++) {
                int d_in_tmp = d + pad_d - kd;
                if (d_in_tmp % stride_d != 0) continue;
                int d_in = d_in_tmp / stride_d;
                if (d_in < 0 || d_in >= in_depth) continue;
                
                #pragma unroll 2
                for (int kh = 0; kh < kH; kh++) {
                    int h_in_tmp = h + pad_h - kh;
                    if (h_in_tmp % stride_h != 0) continue;
                    int h_in = h_in_tmp / stride_h;
                    if (h_in < 0 || h_in >= in_height) continue;
                    
                    #pragma unroll 2
                    for (int kw = 0; kw < kW; kw++) {
                        int w_in_tmp = w + pad_w - kw;
                        if (w_in_tmp % stride_w != 0) continue;
                        int w_in = w_in_tmp / stride_w;
                        if (w_in < 0 || w_in >= in_width) continue;
                        
                        int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * in_height + h_in) * in_width + w_in;
                        int weight_idx = ((((input_channel) * (out_channels / groups) + out_c_local) * kT + kd) * kH + kh) * kW + kw;
                        
                        sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        int output_idx = (((n * out_channels + c) * out_depth + d) * out_height + h) * out_width + w;
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
    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Ensure input tensors are contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value().contiguous();
    }
    
    // Get dimensions
    int N = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(1) * groups;
    int kT = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);
    
    int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];
    
    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());
    
    // Launch configuration
    int threads = 256;
    int blocks = (out_channels * out_depth * out_height * out_width + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_stream_kernel", ([&] {
        // Process input in chunks using different streams
        for (int n = 0; n < N; n += CHUNK_SIZE) {
            int stream_idx = (n / CHUNK_SIZE) % NUM_STREAMS;
            int chunk_size = std::min(CHUNK_SIZE, N - n);
            
            transposed_conv3d_stream_kernel<scalar_t><<<blocks, threads, 0, streams[stream_idx]>>>(
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
                groups,
                n
            );
        }
    }));
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward with stream overlap",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}