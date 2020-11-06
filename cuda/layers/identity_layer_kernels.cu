#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

extern "C" {
#include "layers/identity_layer.h"
#include "dark_cuda.h"
}


extern "C" void forward_identity_layer_gpu(const layer l, network_state net) {
    check_error(cudaMemcpy(l.output_gpu, net.input, l.batch * l.inputs * sizeof(float), cudaMemcpyDeviceToDevice));
}

extern "C" void backward_identity_layer_gpu(layer l, network_state net) {}
