#ifndef CONVOLUTIONAL_LAYER_OPENCV_H
#define CONVOLUTIONAL_LAYER_OPENCV_H

#include <darknet/layers/convolutional_layer.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GPU

void assisted_excitation_forward_gpu_cv(convolutional_layer l, network_state state, int visualizeGroundTruth);

#endif

void assisted_excitation_forward_cv(convolutional_layer l, network_state state, int visualizeGroundTruth);

image *visualize_convolutional_layer(convolutional_layer layer, char *window, image *prev_weights);

#ifdef __cplusplus
}
#endif

#endif
