/* Added on 2019.5.7 */
/* forward only */

#include "layers/linear_layer.h"
#include "utils/blas.h"
#include "assert.h"

layer make_linear_layer(int batch, int h, int w, int c, float a, float b, int verbose) {
    layer l = {(LAYER_TYPE) 0};
    l.type = LINEAR_LAYER;

    l.alpha = a;
    l.beta = b;
    l.groups = c;

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

    l.forward = forward_linear_layer;
    l.backward = backward_linear_layer;
    l.update = update_linear_layer;

#ifdef GPU
    l.forward_gpu = forward_linear_layer_gpu;
    l.backward_gpu = backward_linear_layer_gpu;
    l.update_gpu = update_linear_layer_gpu;

    if (gpu_index >= 0) {
        l.output_gpu = cuda_make_array(l.output, l.batch * l.outputs);
        l.delta_gpu = cuda_make_array(l.delta, l.batch * l.outputs);
    }
#endif

    if (verbose) {
        fprintf(stderr, "lin   a=%4.2f, b=%4.2f     %4d x%4d x%4d -> %4d x%4d x%4d\n", l.alpha, l.beta, l.w, l.h, l.c,
                l.out_w, l.out_h, l.out_c);
    }
    return l;
}

void forward_linear_layer(const layer l, network_state net) {
    // printf("Computing forward linear on CPU");
    axby_cpu(l.batch * l.inputs, l.alpha, l.beta, net.input, 1, l.output, 1);
}

void backward_linear_layer(layer l, network_state net) {
    // TODO
}

void update_linear_layer(layer l, int batch, float learning_rate, float momentum, float decay) {
    // TODO
}

void resize_linear_layer(layer *l, int w, int h) {
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
