/* Added on 2019.5.7 */
#ifndef IDENTITY_LAYER_H
#define IDENTITY_LAYER_H

#include <darknet/layers/layer.h>
#include <darknet/network.h>

#ifdef __cplusplus
extern "C" {
#endif

layer make_identity_layer(int batch, int h, int w, int c, int verbose);

void forward_identity_layer(layer l, network_state net);

void backward_identity_layer(layer l, network_state net);

void resize_identity_layer(layer *l, int w, int h);

#ifdef GPU

void forward_identity_layer_gpu(layer l, network_state net);

void backward_identity_layer_gpu(layer l, network_state net);

#endif

#ifdef __cplusplus
}
#endif

#endif