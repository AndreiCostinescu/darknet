#ifndef REORG_LAYER_H
#define REORG_LAYER_H

#include "images/image.h"
#include "dark_cuda.h"
#include "layers/layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int verbose);
void resize_reorg_layer(layer *l, int w, int h);
void forward_reorg_layer(const layer l, network_state state);
void backward_reorg_layer(const layer l, network_state state);

#ifdef GPU
void forward_reorg_layer_gpu(layer l, network_state state);
void backward_reorg_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif

#endif
