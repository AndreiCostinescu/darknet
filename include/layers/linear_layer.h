/* Added on 2019.5.7 */
#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "layers/layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

layer make_linear_layer(int batch, int h, int w, int c, int a, int b, int verbose);

void forward_linear_layer(layer l, network_state net);

void backward_linear_layer(layer l, network_state net);

void update_linear_layer(layer l, int batch, float learning_rate, float momentum, float decay);

void resize_linear_layer(layer *l, int w, int h);

#ifdef GPU

void forward_linear_layer_gpu(layer l, network_state net);

void backward_linear_layer_gpu(layer l, network_state net);

void update_linear_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay, float loss_scale);

#endif

#ifdef __cplusplus
}
#endif

#endif