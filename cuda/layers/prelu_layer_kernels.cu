#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

extern "C" {
#include "layers/prelu_layer.h"
#include "utils/blas.h"
#include "dark_cuda.h"
}

# if 0

__global__ void forward_prelu_kernel(int n, int w, int h, int c, int g, float *input, float *weights, float *output) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;  // all channel index

    int k = id % c;  // tensor channel index
    float alpha = weights[k / g];

    for (int i = 0; i < h * w; ++i) {
        int idx = i + h * w * id;
        float val = input[idx];
        output[idx] = val < 0 ? alpha * val : val;
    }
}

extern "C" void forward_prelu_layer_gpu(const layer l, network_state net) {
    size_t n = l.c * l.batch;

    forward_prelu_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l.w, l.h, l.c, l.groups, net.input, l.weights_gpu,
                                                      l.output_gpu);
    check_error(cudaPeekAtLastError());
}
#else

__global__ void prelu_kernel(float *input, float *output, float *alpha, int c, int n, int size) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;  // pixel index

    if (offset < size) {
        int filter = blockIdx.y;  // channel index
        int batch = blockIdx.z;  // data index
        int idx_n = filter / n;
        int index = (batch * c + filter) * size + offset;
        float input_val = input[index];
        
        /*
        output[index] = input_val < 0 ? input_val * alpha[idx_n] : input_val;
        /*/
        int condition = input_val >= 0;
        output[index] = input_val * (condition + alpha[idx_n] * (1 - condition));
        //*/
    }
}

extern "C" void prelu_gpu(float *input, float *output, float *alpha, int batch, int c, int n, int size) {
    dim3 dimGrid((size - 1) / BLOCK + 1, c, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    prelu_kernel<<<dimGrid, dimBlock>>>(input, output, alpha, c, n, size);
    check_error(cudaPeekAtLastError());
}

extern "C" void forward_prelu_layer_gpu(const layer l, network_state net) {
    // fill_ongpu(l.outputs * l.batch, 0, l.output_gpu, 1);  // fill output_gpu with 0s
    prelu_gpu(net.input, l.output_gpu, l.weights_gpu, l.batch, l.c, l.groups, l.w * l.h);
}
#endif

extern "C" void backward_prelu_layer_gpu(layer l, network_state net) {
    // TODO
}

extern "C" void update_prelu_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay,
                                       float loss_scale) {
    // TODO
}

extern "C" void pull_prelu_layer(layer l) {
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
}

extern "C" void push_prelu_layer(layer l) {
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
}