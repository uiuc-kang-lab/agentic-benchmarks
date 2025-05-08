#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;
static const int NUM_STREAMS = 4;

template <typename scalar_t>
__global__ void mse_stream_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* partial_sums,
    const int64_t chunk_size
) {
    __shared__ double shm[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double thread_sum = 0.0;

    const int64_t grid_stride = blockDim.x * gridDim.x;
    while (idx < chunk_size) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
        idx += grid_stride;
    }

    shm[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(partial_sums + blockIdx.z, shm[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "Predictions must be CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "Targets must be CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(), "Input sizes must match");

    const int64_t num_elements = predictions.numel();
    auto options = predictions.options().dtype(at::kDouble);
    auto final_accumulator = torch::zeros({1}, options);

    // Setup streams and partial sums
    cudaStream_t streams[NUM_STREAMS];
    torch::Tensor partials = torch::zeros({NUM_STREAMS}, options);
    double* partials_ptr = partials.data_ptr<double>();

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    const int64_t chunk_size = (num_elements + NUM_STREAMS - 1) / NUM_STREAMS;
    
    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_async_forward", [&] {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            const int64_t start = i * chunk_size;
            const int64_t end = std::min((i+1)*chunk_size, num_elements);
            const int elements_this_stream = end - start;
            
            if (elements_this_stream <= 0) continue;

            const int grid_size = (elements_this_stream + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            mse_stream_kernel<scalar_t><<<grid_size, BLOCK_SIZE, 0, streams[i]>>>(
                predictions.data_ptr<scalar_t>() + start,
                targets.data_ptr<scalar_t>() + start,
                partials_ptr,
                elements_this_stream
            );
        }
    });

    // Sync and reduce partial results
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    final_accumulator += partials.sum().div_(num_elements);
    return final_accumulator.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MSE loss with stream-parallel computation");
}
