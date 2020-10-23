/* Added on 2019.5.7 */
#ifndef PRELU_LAYER_H
#define PRELU_LAYER_H

#include "layers/layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

layer make_prelu_layer(int batch, int h, int w, int c, int n, int verbose);

void forward_prelu_layer(layer l, network_state net);

void backward_prelu_layer(layer l, network_state net);

void update_prelu_layer(layer l, int batch, float learning_rate, float momentum, float decay);

void resize_prelu_layer(layer *l, int w, int h);

#ifdef GPU

void pull_prelu_layer(layer l);

void push_prelu_layer(layer l);

void forward_prelu_layer_gpu(layer l, network_state net);

void backward_prelu_layer_gpu(layer l, network_state net);

void update_prelu_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay, float loss_scale);

#endif

#ifdef __cplusplus
}
#endif

#endif