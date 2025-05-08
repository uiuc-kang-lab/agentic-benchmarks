#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__constant__ float c_weight[16384];

#define NUM_STREAMS 4
#define BLOCK_SIZE 256

__global__ void conv_transpose2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N,
    const int C_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int H_out,
    const int W_out,
    const int kH,
    const int kW,
    const int sH,
    const int sW,
    const int pH,
    const int pW,
    const int batch_offset,
    const int batch_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * C_out * H_out * W_out) return;

    const int ow = idx % W_out;
    const int oh = (idx / W_out) % H_out;
    const int oc = (idx / (W_out * H_out)) % C_out;
    const int n = batch_offset + (idx / (W_out * H_out * C_out));

    float sum = 0.0f;

    #pragma unroll
    for (int ic = 0; ic < C_in; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < kH; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kW; ++kw) {
                const int i_val = oh + pH - kh;
                const int j_val = ow + pW - kw;

                if ((i_val % sH == 0) && (j_val % sW == 0)) {
                    const int i_in = i_val / sH;
                    const int j_in = j_val / sW;

                    if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                        const int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                        const int weight_idx = ((ic * C_out + oc) * kH + kh) * kW + kw;
                        sum += input[input_idx] * c_weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    const int output_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
    output[output_idx] = sum;
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    const int weight_size = weight.numel() * sizeof(float);
    if (weight_size > 64 * 1024) {
        c10::optional<torch::Tensor> bias = c10::nullopt;
        if (!bias_obj.is_none()) {
            bias = bias_obj.cast<torch::Tensor>();
        }
        return at::conv_transpose2d(x, weight, bias, stride, padding);
    }

    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), weight_size);

    torch::Tensor bias;
    const float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
        bias_ptr = bias.data_ptr<float>();
    }

    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);
    const int C_out = weight.size(1);
    const int kH = weight.size(2);
    const int kW = weight.size(3);
    const int sH = stride[0];
    const int sW = stride[1];
    const int pH = padding[0];
    const int pW = padding[1];

    const int H_out = (H_in - 1) * sH - 2 * pH + kH;
    const int W_out = (W_in - 1) * sW - 2 * pW + kW;

    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate batch size per stream
    const int batch_per_stream = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    const int elements_per_batch = C_out * H_out * W_out;

    // Process batches in parallel streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        const int batch_offset = i * batch_per_stream;
        const int current_batch_size = std::min(batch_per_stream, N - batch_offset);
        
        if (current_batch_size <= 0) continue;

        const int total_elements = current_batch_size * elements_per_batch;
        const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Launch kernel in stream
        conv_transpose2d_forward_kernel<<<num_blocks, BLOCK_SIZE, 0, streams[i]>>>(
            x.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            kH, kW, sH, sW, pH, pW,
            batch_offset,
            current_batch_size
        );
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with stream pipelining",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}