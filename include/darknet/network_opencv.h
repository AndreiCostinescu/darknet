#ifndef NETWORK_OPENCV_H
#define NETWORK_OPENCV_H

#include <darknet/darknet.h>
#include <darknet/network.h>
#include <darknet/utils/data.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GPU

float train_network_datum_gpu_cv(network net, float *x, float *y);

void backward_network_gpu_cv(network net, network_state state);

#endif

void visualize_network(network net);

float train_network_waitkey(network net, data d, int wait_key);

#ifdef __cplusplus
}
#endif

#endif
