#ifndef DECONVOLUTIONAL_LAYER_H
#define DECONVOLUTIONAL_LAYER_H

#include <darknet/dark_cuda.h>
#include <darknet/images/image.h>
#include <darknet/layers/activations.h>
#include <darknet/layers/layer.h>
#include <darknet/network.h>

typedef layer deconvolutional_layer;

#ifdef __cplusplus
extern "C" {
#endif
#ifdef GPU
void forward_deconvolutional_layer_gpu(deconvolutional_layer layer, network_state state);
void backward_deconvolutional_layer_gpu(deconvolutional_layer layer, network_state state);
void update_deconvolutional_layer_gpu(deconvolutional_layer layer, int skip, float learning_rate, float momentum,
                                      float decay);
void push_deconvolutional_layer(deconvolutional_layer layer);
void pull_deconvolutional_layer(deconvolutional_layer layer);
#endif

deconvolutional_layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride,
                                                 ACTIVATION activation, int verbose);
void resize_deconvolutional_layer(deconvolutional_layer *layer, int h, int w);
void forward_deconvolutional_layer(const deconvolutional_layer layer, network_state state);
void update_deconvolutional_layer(deconvolutional_layer layer, int skip, float learning_rate, float momentum,
                                  float decay);
void backward_deconvolutional_layer(deconvolutional_layer layer, network_state state);

image get_deconvolutional_image(deconvolutional_layer layer);
image get_deconvolutional_delta(deconvolutional_layer layer);
image get_deconvolutional_filter(deconvolutional_layer layer, int i);

int deconvolutional_out_height(deconvolutional_layer layer);
int deconvolutional_out_width(deconvolutional_layer layer);

#ifdef __cplusplus
}
#endif

#endif
