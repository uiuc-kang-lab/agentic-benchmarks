#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define a vectorized structure for coalesced memory access
struct __align__(16) Float4 {
    float x, y, z, w;
};

// Combined kernel that processes the prediction-target arrays using vectorized loads for most elements
// and a grid-stride loop to gracefully handle any remainder elements in a single launch.
__global__ void hinge_loss_combined_kernel(const float* __restrict__ predictions,
                                            const float* __restrict__ targets,
                                            float* __restrict__ output,
                                            int n4,       // number of groups of 4 elements
                                            int remainder // number of leftover elements
                                            ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Process main vectorized portion using Float4 loads/stores
    for (int i = tid; i < n4; i += totalThreads) {
        // Load 4 elements at a time
        Float4 pred4 = *((const Float4*)(predictions) + i);
        Float4 targ4 = *((const Float4*)(targets) + i);

        // Compute hinge loss for each component
        Float4 out4;
        out4.x = fmaxf(0.0f, 1.0f - pred4.x * targ4.x);
        out4.y = fmaxf(0.0f, 1.0f - pred4.y * targ4.y);
        out4.z = fmaxf(0.0f, 1.0f - pred4.z * targ4.z);
        out4.w = fmaxf(0.0f, 1.0f - pred4.w * targ4.w);

        // Store the results
        *((Float4*)(output) + i) = out4;
    }

    // Process any remaining elements
    int offset = n4 * 4;
    for (int i = tid; i < remainder; i += totalThreads) {
        int idx = offset + i;
        float pred = __ldg(&predictions[idx]);
        float targ = __ldg(&targets[idx]);
        output[idx] = fmaxf(0.0f, 1.0f - pred * targ);
    }
}

// Forward function utilizing the combined kernel
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    int n4 = n / 4;      // bulk of data processed in groups of 4
    int remainder = n % 4; // leftover single elements
    
    torch::Tensor output = torch::empty_like(predictions);

    // Determine grid dimensions based on the larger workload among vectorized and remainder parts
    int workload = (n4 > remainder) ? n4 : remainder;
    int threads = 256;
    int blocks = (workload + threads - 1) / threads;

    hinge_loss_combined_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n4,
        remainder
    );

    // Compute the mean of the output tensor on the host
    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined Vectorized Hinge Loss Forward");
}
