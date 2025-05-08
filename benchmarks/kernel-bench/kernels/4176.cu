#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <algorithm>

// Combined kernel using grid-stride loops. The kernel processes a sub-range [start, end) of the input.
template <typename scalar_t>
__global__ void hardtanh_kernel(const scalar_t* __restrict__ x,
                                  scalar_t* __restrict__ out,
                                  int64_t start,
                                  int64_t end,
                                  scalar_t min_val,
                                  scalar_t max_val) {
    // Compute global index starting from the offset 'start'
    int64_t idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    // Use grid-stride loop to cover the assigned segment
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < end; idx += stride) {
        scalar_t val = x[idx];
        // Clamp val between min_val and max_val
        if (val < min_val) {
            val = min_val;
        } else if (val > max_val) {
            val = max_val;
        }
        out[idx] = val;
    }
}

// Hybrid forward function: uses a single stream for small tensors, and multi-stream for large tensors
at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();
    const int threads = 1024;

    // Decide on multi-stream usage based on tensor size
    // For very large tensors, overlapping kernel launches on separate streams may hide latency
    const bool use_multistream = (numel > 100000);

    if (!use_multistream) {
        // Single-stream: launch one kernel over the entire tensor with grid-stride loop
        int blocks = (numel + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
            hardtanh_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                0, numel,
                static_cast<scalar_t>(min_val),
                static_cast<scalar_t>(max_val)
            );
        }));
    } else {
        // Multi-stream: split the data into segments processed in parallel
        const int num_streams = 4;
        std::vector<cudaStream_t> streams(num_streams);
        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreate(&streams[i]);
        }

        int64_t segment_size = (numel + num_streams - 1) / num_streams;

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
            for (int i = 0; i < num_streams; i++) {
                int64_t start = i * segment_size;
                if (start >= numel) break; // no more work
                int64_t end = std::min(start + segment_size, numel);
                int blocks = ((end - start) + threads - 1) / threads;
                hardtanh_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                    x.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    start,
                    end,
                    static_cast<scalar_t>(min_val),
                    static_cast<scalar_t>(max_val)
                );
            }
        }));

        // Synchronize and destroy streams
        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);
        }
    }
    return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
    if (!x.is_cuda()) {
        throw std::invalid_argument("Input tensor must be a CUDA tensor");
    }
    return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized HardTanh Activation (CUDA) using hybrid streams and grid-stride loops");
}
