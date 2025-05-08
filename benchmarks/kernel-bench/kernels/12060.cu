#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Vector type for coalesced memory access
typedef float4 vec4_t;

__device__ __forceinline__ float process_element(float pred, float target) {
    return fmaxf(0.0f, 1.0f - pred * target);
}

__global__ void hinge_loss_coalesced_kernel(const float* __restrict__ predictions,
                                          const float* __restrict__ targets,
                                          float* __restrict__ output,
                                          const int n) {
    // Calculate aligned size for vectorized processing
    const int vec_size = 4;
    const int aligned_n = n & ~(vec_size - 1);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Process 4 elements at a time using vector loads
    for (int i = tid * vec_size; i < aligned_n; i += stride * vec_size) {
        // Vector loads
        vec4_t pred_vec = *reinterpret_cast<const vec4_t*>(predictions + i);
        vec4_t target_vec = *reinterpret_cast<const vec4_t*>(targets + i);

        // Process each component
        float4 result;
        result.x = process_element(pred_vec.x, target_vec.x);
        result.y = process_element(pred_vec.y, target_vec.y);
        result.z = process_element(pred_vec.z, target_vec.z);
        result.w = process_element(pred_vec.w, target_vec.w);

        // Coalesced write
        *reinterpret_cast<vec4_t*>(output + i) = result;
    }

    // Handle remaining elements
    for (int i = aligned_n + tid; i < n; i += stride) {
        const float pred = __ldg(&predictions[i]);
        const float target = __ldg(&targets[i]);
        output[i] = process_element(pred, target);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 65535);

    hinge_loss_coalesced_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Memory Hinge Loss Forward");
}