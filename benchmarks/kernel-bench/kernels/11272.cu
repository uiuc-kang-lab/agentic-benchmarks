#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int start_idx,
    int chunk_size
) {
    const int elements_per_thread = 4;
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * blockDim.x * elements_per_thread;
    const int idx_start = start_idx + block_offset + tid * elements_per_thread;
    float thread_sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < elements_per_thread; j++) {
        int idx = idx_start + j;
        if (idx >= start_idx + chunk_size) break;
        
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
    }

    __shared__ float shared_sum[blockDim.x];
    shared_sum[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared_sum[0]);
    }
}

torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(
        predictions.sizes() == targets.sizes(),
        "Input tensors must have the same shape"
    );
    TORCH_CHECK(
        predictions.is_contiguous() && targets.is_contiguous(),
        "Input tensors must be contiguous"
    );
    TORCH_CHECK(
        predictions.device().is_cuda() && targets.device().is_cuda(),
        "Inputs must be CUDA tensors"
    );

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int num_streams = 4;
    const int elements_per_stream = (n + num_streams - 1) / num_streams;
    
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int block_size = 256;
    for (int i = 0; i < num_streams; ++i) {
        int start = i * elements_per_stream;
        int valid_size = min(elements_per_stream, n - start);
        if (valid_size <= 0) break;

        int grid_size = (valid_size + block_size * 4 - 1) / (block_size * 4);
        smooth_l1_loss_kernel<<<grid_size, block_size, 0, streams[i]>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            start,
            valid_size
        );
    }

    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    output.div_(n);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss (CUDA)");
}