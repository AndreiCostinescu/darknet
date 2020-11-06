/* Added on 2019.5.7 */
/* forward only */

#include "layers/prelu_layer.h"
#include "utils/blas.h"
#include "assert.h"

layer make_prelu_layer(int batch, int h, int w, int c, int n, int verbose) {
    layer l = {(LAYER_TYPE) 0};
    l.type = PRELU;

    assert(c % n == 0);

    l.n = n;
    l.groups = c / n;   // the number of feature maps in a group

    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.inputs = h * w * c;

    l.out_h = h;
    l.out_w = w;
    l.out_c = c;
    l.outputs = h * w * c;

    l.output = static_cast<float *>(xcalloc(batch * l.outputs, sizeof(float)));
    l.delta = static_cast<float *>(xcalloc(batch * l.outputs, sizeof(float)));

    l.weights = static_cast<float *>(xcalloc(n, sizeof(float)));
    l.weight_updates = static_cast<float *>(xcalloc(n, sizeof(float)));

    l.nweights = n;

    l.forward = forward_prelu_layer;
    l.backward = backward_prelu_layer;
    l.update = update_prelu_layer;

#ifdef GPU
    l.forward_gpu = forward_prelu_layer_gpu;
    l.backward_gpu = backward_prelu_layer_gpu;
    l.update_gpu = update_prelu_layer_gpu;

    if (gpu_index >= 0) {
        l.output_gpu = cuda_make_array(l.output, l.batch * l.outputs);
        l.delta_gpu = cuda_make_array(l.delta, l.batch * l.outputs);
        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
    }
#endif

    if (verbose) {
        fprintf(stderr, "prelu   %3d                  %4d x%4d x%4d -> %4d x%4d x%4d\n", n, l.w, l.h, l.c, l.out_w,
                l.out_h, l.out_c);
    }
    return l;
}

void forward_prelu_layer(const layer l, network_state net) {
    printf("Computing forward prelu on CPU");
    int size = l.w * l.h * l.groups, startIndex, mask;
    float alpha, inputValue;
    for (int i = 0; i < l.batch; i++) {
        for (int j = 0; j < l.n; j++) {
            alpha = l.weights[j];
            startIndex = i * l.outputs + j * size;
            for (int k = 0; k < size; k++) {
                inputValue = net.input[startIndex + k];
                mask = (int) (inputValue > 0);
                l.output[startIndex + k] = (float) mask * inputValue + (float) (1 - mask) * alpha * inputValue;
            }
        }
    }
}

void backward_prelu_layer(layer l, network_state net) {
    // TODO
}

void update_prelu_layer(layer l, int batch, float learning_rate, float momentum, float decay) {
    // TODO
}

void resize_prelu_layer(layer *l, int w, int h) {
    l->w = w;
    l->h = h;
    l->inputs = w * h * l->c;

    l->out_w = w;
    l->out_h = h;
    l->outputs = w * h * l->out_c;

    l->output = static_cast<float *>(xrealloc(l->output, l->batch * l->outputs * sizeof(float)));
    l->delta = static_cast<float *>(xrealloc(l->delta, l->batch * l->outputs * sizeof(float)));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);
    l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
#endif

}
