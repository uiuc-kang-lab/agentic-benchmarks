#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Balanced kernel that evenly distributes workloads across threads

template <typename scalar_t>
__global__ void balanced_tanh_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    // For float type, use vectorized processing with float4
    if (sizeof(scalar_t) == sizeof(float)) {
        int vec4_count = size / 4;                // Number of groups of 4
        int remainder_count = size - vec4_count * 4;  // Remaining elements after groups

        // Evenly divide the vectorized workload among threads
        int chunk_v = (vec4_count + totalThreads - 1) / totalThreads;  // Ceiling division
        int start_v = tid * chunk_v;
        int end_v = start_v + chunk_v;
        if(end_v > vec4_count) end_v = vec4_count;

        const float4* input4 = reinterpret_cast<const float4*>(input);
        float4* output4 = reinterpret_cast<float4*>(output);

        for (int i = start_v; i < end_v; i++) {
            float4 in_val = input4[i];
            float4 out_val;
            out_val.x = tanhf(in_val.x);
            out_val.y = tanhf(in_val.y);
            out_val.z = tanhf(in_val.z);
            out_val.w = tanhf(in_val.w);
            output4[i] = out_val;
        }

        // Evenly distribute remaining scalar elements
        int rem_start = vec4_count * 4;
        int chunk_r = (remainder_count + totalThreads - 1) / totalThreads;
        int start_r = tid * chunk_r;
        int end_r = start_r + chunk_r;
        if(end_r > remainder_count) end_r = remainder_count;

        for (int i = rem_start + start_r; i < rem_start + end_r; i++) {
            output[i] = tanhf(input[i]);
        }

    } else {
        // Fallback scalar loop for types other than float (e.g., double)
        int chunk = (size + totalThreads - 1) / totalThreads;
        int start = tid * chunk;
        int end = start + chunk;
        if(end > size) end = size;
        for (int i = start; i < end; i++) {
            output[i] = tanh(input[i]);
        }
    }
}


torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    int blocks = 0;

    // For float use vectorized workload in terms of groups of 4 elements
    if (input.scalar_type() == torch::kFloat32) {
        int vec4_count = size / 4;
        blocks = (vec4_count > 0) ? ((vec4_count + threads - 1) / threads) : 1;
    } else {
        blocks = (size + threads - 1) / threads;
    }

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "balanced_tanh_kernel", ([&] {
        balanced_tanh_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced vectorized Tanh forward (CUDA)");
}
