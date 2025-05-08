#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int BLOCK_SIZE = 256;
const int NUM_STREAMS = 4;  // Number of concurrent streams

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_offset,
    const int N_chunk,
    const int C_in,
    const int D_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int outD,
    const int outH,
    const int outW,
    const int groups,
    const int in_channels_per_group) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_chunk * C_in * D_in * H_in * W_in;
    if (index >= total) return;

    // Decode the flattened index
    int w = index % W_in;
    int h = (index / W_in) % H_in;
    int d = (index / (W_in * H_in)) % D_in;
    int c_in = (index / (W_in * H_in * D_in)) % C_in;
    int n_local = index / (W_in * H_in * D_in * C_in);
    int n = n_local + batch_offset;  // Global batch index

    int group = c_in / in_channels_per_group;
    float inp_val = input[index];

    // Pre-compute output base indices
    int out_d_base = d * stride_d - pad_d;
    int out_h_base = h * stride_h - pad_h;
    int out_w_base = w * stride_w - pad_w;

    // Shared memory for frequently accessed values
    __shared__ int shared_stride[3];
    if (threadIdx.x < 3) {
        shared_stride[0] = outH * outW;
        shared_stride[1] = outW;
        shared_stride[2] = 1;
    }
    __syncthreads();

    for (int kd = 0; kd < kernel_d; kd++) {
        int out_d = out_d_base + kd;
        if (out_d < 0 || out_d >= outD) continue;

        for (int kh = 0; kh < kernel_h; kh++) {
            int out_h = out_h_base + kh;
            if (out_h < 0 || out_h >= outH) continue;

            for (int kw = 0; kw < kernel_w; kw++) {
                int out_w = out_w_base + kw;
                if (out_w < 0 || out_w >= outW) continue;

                // Compute weight offset once per kernel position
                int kernel_offset = ((kd * kernel_h + kh) * kernel_w + kw) * (C_out / groups);

                #pragma unroll 4
                for (int oc = 0; oc < (C_out / groups); oc++) {
                    int weight_idx = (c_in * (C_out / groups) * kernel_d * kernel_h * kernel_w) + 
                                   kernel_offset + oc;
                    float weight_val = weight[weight_idx];
                    
                    int oc_global = group * (C_out / groups) + oc;
                    int out_idx = (((n * C_out + oc_global) * outD + out_d) * outH + out_h) * outW + out_w;
                    
                    atomicAdd(&output[out_idx], inp_val * weight_val);
                }
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) CHECK_INPUT(*bias);

    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);
    
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    const int C_out = weight.size(1) * groups;

    const int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
    const int pad_d = padding[0], pad_h = padding[1], pad_w = padding[2];
    const int out_pad_d = output_padding[0], out_pad_h = output_padding[1], out_pad_w = output_padding[2];

    const int outD = (D_in - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
    const int outH = (H_in - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
    const int outW = (W_in - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;

    auto output = torch::zeros({N, C_out, outD, outH, outW}, input.options());

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int batch_per_stream = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    const int elements_per_batch = C_in * D_in * H_in * W_in;

    for (int stream_idx = 0; stream_idx < NUM_STREAMS; stream_idx++) {
        int batch_start = stream_idx * batch_per_stream;
        int batch_end = std::min(batch_start + batch_per_stream, N);
        if (batch_start >= N) break;

        int N_chunk = batch_end - batch_start;
        int total_elements = N_chunk * elements_per_batch;
        
        const float* input_ptr = input.data_ptr<float>() + batch_start * elements_per_batch;
        float* output_ptr = output.data_ptr<float>();
        const float* weight_ptr = weight.data_ptr<float>();

        int threads = BLOCK_SIZE;
        int blocks = (total_elements + threads - 1) / threads;

        conv_transpose3d_kernel<<<blocks, threads, 0, streams[stream_idx]>>>(
            input_ptr, weight_ptr, output_ptr,
            batch_start, N_chunk, C_in, D_in, H_in, W_in,
            C_out, kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            outD, outH, outW,
            groups, C_in / groups);
    }

    // Synchronize all streams
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    if (bias.has_value()) {
        output.add_(*bias).view({1, -1, 1, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed Conv3D forward (CUDA) with streams");
}