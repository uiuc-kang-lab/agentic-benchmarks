#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int num_features,
    const float eps
) {
    const int batch_id = blockIdx.x;
    const int feature_start = threadIdx.x * 4;
    const int feature_end = min(feature_start + 4, num_features);
    
    // Calculate unique element based on batch and feature indices
    for (int batch_offset = batch_id * num_features;
         batch_offset < (batch_id + 1) * num_features;
         batch_offset += num_features) {
        
        // Calculate sum of squares
        scalar_t sumsq = 0.0f;  
        for (int feat = feature_start; feat < feature_end; ++feat) {
            if (feat < num_features) {
                scalar_t val = input[batch_offset + feat];
                sumsq += val * val;
            }
        }
        
        // Calculate RMS
        const scalar_t rms = sqrt(sumsq / num_features + eps);

        // Normalize
        for (int feat = feature_start; feat < feature_end; ++feat) {
            if (feat < num_features) {
                output[batch_offset + feat] = input[batch_offset + feat] / rms;
            }
        }
    }
}

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    const int threads_per_block = 64; // assuming multiple of 4 for coalesced memory access

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<batch_size, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_features,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA)");
}
