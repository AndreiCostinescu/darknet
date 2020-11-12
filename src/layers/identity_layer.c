/* Added on 2019.5.7 */
/* forward only */

#include <darknet/layers/identity_layer.h>
#include <darknet/utils/blas.h>
#include <assert.h>

layer make_identity_layer(int batch, int h, int w, int c, int verbose) {
    layer l = {(LAYER_TYPE) 0};
    l.type = IDENTITY;

    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.inputs = h * w * c;

    l.out_h = h;
    l.out_w = w;
    l.out_c = c;
    l.outputs = h * w * c;

    l.output = static_cast<float *>(xcalloc(l.batch * l.outputs, sizeof(float)));
    l.delta = static_cast<float *>(xcalloc(l.batch * l.outputs, sizeof(float)));

    l.forward = forward_identity_layer;
    l.backward = backward_identity_layer;

#ifdef GPU
    l.forward_gpu = forward_identity_layer_gpu;
    l.backward_gpu = backward_identity_layer_gpu;

    if (gpu_index >= 0) {
        l.output_gpu = cuda_make_array(l.output, l.batch * l.outputs);
        l.delta_gpu = cuda_make_array(l.delta, l.batch * l.outputs);
    }
#endif

    if (verbose) {
        fprintf(stderr, "identity                   %4d x%4d x%4d   ->  %4d x%4d x%4d\n", l.w, l.h, l.c, l.out_w,
                l.out_h, l.out_c);
    }
    return l;
}

void forward_identity_layer(layer l, network_state net) {
    printf("Computing forward identity on CPU");
    memcpy(l.output, net.input, l.batch * l.inputs * sizeof(float));
}

void backward_identity_layer(layer l, network_state net) {}

void resize_identity_layer(layer *l, int w, int h) {
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
