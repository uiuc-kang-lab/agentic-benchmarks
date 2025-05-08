#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel for pointwise 2D convolution operating on a batch chunk
__global__ void pipelined_conv_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int B,    // batch size for this chunk
    int IC,
    int OC,
    int H,
    int W
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * OC * H * W;
    if (index >= total_elements) return;

    int w = index % W;
    int h = (index / W) % H;
    int oc = (index / (W * H)) % OC;
    int b = index / (W * H * OC);

    float sum = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        int x_idx = b * (IC * H * W) + ic * (H * W) + h * W + w;
        int w_idx = oc * IC + ic;
        sum += x[x_idx] * weight[w_idx];
    }
    
    if (bias)
        output[index] = sum + bias[oc];
    else
        output[index] = sum;
}

// This function uses CUDA streams to overlap computation across different batch chunks
// enabling pipelining of memory operations (if any) along with kernel computations.
// The input batch is partitioned into several chunks; each chunk is processed asynchronously
// on its own stream. This strategy is beneficial especially on hardware supporting concurrent
// kernel execution.

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (B, IC, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias_opt) {
        TORCH_CHECK(bias_opt->is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias_opt->dim() == 1, "Bias must be 1D");
    }

    int B = x.size(0);
    int IC = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int OC = weight.size(0);

    TORCH_CHECK(weight.size(1) == IC, "Input/output channel mismatch");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "Kernel must be 1x1");
    if (bias_opt) {
        TORCH_CHECK(bias_opt->size(0) == OC, "Bias/out channel mismatch");
    }

    // Create output tensor
    auto output = torch::empty({B, OC, H, W}, x.options());

    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias_opt ? bias_opt->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    const int threads = 256;

    // Create several CUDA streams for pipelining. Use up to 4 streams or fewer if B is small.
    int num_streams = (B < 4) ? B : 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Partition the batch among the streams
    int chunk_size = (B + num_streams - 1) / num_streams; // Ceiling division
    for (int i = 0; i < num_streams; ++i) {
        int start_b = i * chunk_size;
        if (start_b >= B) break;  // no more chunks
        int end_b = std::min(start_b + chunk_size, B);
        int B_chunk = end_b - start_b;

        int total_elements = B_chunk * OC * H * W;
        int blocks = (total_elements + threads - 1) / threads;

        // Offset the input and output pointers for the current batch chunk
        const float* x_chunk_ptr = x_ptr + start_b * (IC * H * W);
        float* out_chunk_ptr = out_ptr + start_b * (OC * H * W);

        // Launch kernel asynchronously on the designated stream
        pipelined_conv_kernel<<<blocks, threads, 0, streams[i]>>>(
            x_chunk_ptr, w_ptr, b_ptr, out_chunk_ptr,
            B_chunk, IC, OC, H, W
        );
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pipelined Conv 2D convolution forward (CUDA)");
}
