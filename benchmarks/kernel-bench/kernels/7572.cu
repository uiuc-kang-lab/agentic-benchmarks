#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16
#define NUM_STREAMS 4
#define WARP_SIZE 32

__global__ void conv_transpose3d_streamed_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_offset,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int iD, const int iH, const int iW,
    const int kD, const int kH, const int kW,
    const int outD, const int outH, const int outW,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int groups) {

    extern __shared__ float shared_weight[];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int out_x = bx * TILE_SIZE + tx;
    const int out_y = by * TILE_SIZE + ty;
    
    if (out_x >= outW || out_y >= outH) return;
    
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    
    for (int b = batch_offset; b < batch_offset + batch_size; b++) {
        for (int oc = threadIdx.z; oc < out_channels; oc += blockDim.z) {
            const int group = oc / out_channels_per_group;
            const int oc_within_group = oc % out_channels_per_group;
            
            for (int od = 0; od < outD; od++) {
                float sum = 0.0f;
                
                const int weight_load_idx = tx + ty * TILE_SIZE + threadIdx.z * TILE_SIZE * TILE_SIZE;
                if (weight_load_idx < kD * kH * kW * in_channels_per_group) {
                    shared_weight[weight_load_idx] = weight[weight_load_idx + oc * kD * kH * kW * in_channels_per_group];
                }
                __syncthreads();
                
                const int in_d_start = (od + pad_d) / stride_d;
                const int in_h_start = (out_y + pad_h) / stride_h;
                const int in_w_start = (out_x + pad_w) / stride_w;
                
                for (int ic = group * in_channels_per_group; ic < (group + 1) * in_channels_per_group; ic++) {
                    #pragma unroll 4
                    for (int kd = 0; kd < kD; kd++) {
                        const int id = in_d_start - kd;
                        if (id < 0 || id >= iD) continue;
                        
                        #pragma unroll 4
                        for (int kh = 0; kh < kH; kh++) {
                            const int ih = in_h_start - kh;
                            if (ih < 0 || ih >= iH) continue;
                            
                            #pragma unroll 4
                            for (int kw = 0; kw < kW; kw++) {
                                const int iw = in_w_start - kw;
                                if (iw < 0 || iw >= iW) continue;
                                
                                const int input_idx = ((b * in_channels + ic) * iD + id) * iH * iW + ih * iW + iw;
                                const int weight_idx = ((ic % in_channels_per_group) * kD + kd) * kH * kW + kh * kW + kw;
                                
                                sum += input[input_idx] * shared_weight[weight_idx];
                            }
                        }
                    }
                }
                
                if (bias != nullptr) {
                    sum += bias[oc];
                }
                
                const int out_idx = ((b * out_channels + oc) * outD + od) * outH * outW + out_y * outW + out_x;
                output[out_idx] = sum;
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {
    
    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int iD = x.size(2);
    const int iH = x.size(3);
    const int iW = x.size(4);
    
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);
    
    const int out_channels = groups * weight.size(1);
    
    const int stride_d = stride[0];
    const int stride_h = stride[1];
    const int stride_w = stride[2];
    
    const int pad_d = padding[0];
    const int pad_h = padding[1];
    const int pad_w = padding[2];
    
    const int outD = (iD - 1) * stride_d - 2 * pad_d + kD + output_padding[0];
    const int outH = (iH - 1) * stride_h - 2 * pad_h + kH + output_padding[1];
    const int outW = (iW - 1) * stride_w - 2 * pad_w + kW + output_padding[2];
    
    auto output = torch::zeros({batch, out_channels, outD, outH, outW}, x.options());
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int batch_per_stream = (batch + NUM_STREAMS - 1) / NUM_STREAMS;
    
    dim3 threads(TILE_SIZE, TILE_SIZE, 4);
    dim3 blocks((outW + TILE_SIZE - 1) / TILE_SIZE,
                (outH + TILE_SIZE - 1) / TILE_SIZE);
    
    const int shared_memory_size = kD * kH * kW * (in_channels / groups) * sizeof(float);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        const int batch_offset = i * batch_per_stream;
        const int current_batch_size = min(batch_per_stream, batch - batch_offset);
        
        if (current_batch_size <= 0) continue;
        
        conv_transpose3d_streamed_kernel<<<blocks, threads, shared_memory_size, streams[i]>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_offset,
            current_batch_size,
            in_channels,
            out_channels,
            iD, iH, iW,
            kD, kH, kW,
            outD, outH, outW,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            groups
        );
    }
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed ConvTranspose3d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}