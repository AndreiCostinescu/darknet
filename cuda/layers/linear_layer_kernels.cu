#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

extern "C" {
#include "layers/linear_layer.h"
#include "utils/blas.h"
#include "dark_cuda.h"
}

extern "C" void forward_linear_layer_gpu(const layer l, network_state net) {
    int N = l.batch * l.inputs;
    // printData(net.input, N, "LINEAR LAYER INPUTS");
    // printf("alpha = %f, beta = %f\n", l.alpha, l.beta);
    axby_ongpu(N, l.alpha, l.beta, net.input, 1, l.output_gpu, 1);
    // printData(l.output_gpu, N, "LINEAR LAYER OUTPUTS");
}

extern "C" void backward_linear_layer_gpu(layer l, network_state net) {
    // TODO
}

extern "C" void update_linear_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay,
                                        float loss_scale) {
    // TODO
}
