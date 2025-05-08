#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename T>
struct alignas(16) Vec4 { T x, y, z, w; };

__global__ void hinge_loss_kernel(const float* __restrict__ predictions,
                                 const float* __restrict__ targets,
                                 float* __restrict__ output,
                                 int n) {
    const int vec_size = 4;
    const int vec_n = n / vec_size;
    const int remainder = n % vec_size;

    // Vectorized processing
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < vec_n;
        i += blockDim.x * gridDim.x) {
        auto pred = *reinterpret_cast<const Vec4<float>*>(&predictions[i*vec_size]);
        auto tgt = *reinterpret_cast<const Vec4<float>*>(&targets[i*vec_size]);
        Vec4<float> result;
        
        #pragma unroll
        for(int j = 0; j < vec_size; ++j) {
            float val = 1.0f - reinterpret_cast<const float*>(&pred)[j] * __ldg(&reinterpret_cast<const float*>(&tgt)[j]);
            reinterpret_cast<float*>(&result)[j] = val > 0.0f ? val : 0.0f;
        }
        *reinterpret_cast<Vec4<float>*>(&output[i*vec_size]) = result;
    }

    // Remainder elements
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int base_idx = vec_n * vec_size;
    if(tid < remainder) {
        const int idx = base_idx + tid;
        float pred = __ldg(&predictions[idx]);
        float tgt = __ldg(&targets[idx]);
        output[idx] = fmaxf(0.0f, 1.0f - pred * tgt);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    const int threads = 256;
    const int vec_threads = (n / 4 + threads - 1) / threads;
    int blocks = min(65535, vec_threads);

    hinge_loss_kernel<<<blocks, threads>>>(predictions.data_ptr<float>(),
                                         targets.data_ptr<float>(),
                                         output.data_ptr<float>(),
                                         n);

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward");
}